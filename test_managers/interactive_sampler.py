import numpy as np
import tkinter
import tkinter.ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageOps

import torch

from latent_sampler import LatentSampler
from utils import manually_seed


class TkinterProgressBarWrapper():
    """
    A tkinter pbar wrapper that simulates tqdm behavior
    """
    def __init__(self, root, tkinter_pbar):
        self.root = root
        self.pbar = tkinter_pbar

        self.counter = 0
        self.length = 0
        self.iter_target = None

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        percentage = self.counter / self.length
        self.pbar['value'] = int(round(percentage*100))
        self.pbar.update()
        return next(self.iter_target)
    
    def __call__(self, target):
        self.counter = 0
        self.length = len(target)
        self.iter_target = iter(target)
        self.pbar.stop()
        self.pbar['value'] = 0
        self.pbar.update()
        return self

class InteractiveSampler():

    def __init__(self, task_manager, testing_vars, config):

        assert config.task.batch_size == 1, \
            "Bro, you have only one mouse, why you need a batch more than one image???"

        # In case the image is too large, setting this value to N will downsample your image by N times
        self.fov_rescale = 1

        self.task_manager = task_manager
        self.g_ema_module = task_manager.g_ema_module
        self.latent_sampler = LatentSampler(self.g_ema_module, config)

        self.root = tkinter.Tk()
        self.root.title('Interactive InfinityGAN')
        self.root.iconphoto(False, tkinter.PhotoImage(file="assets/favicon.ico"))

        # Create canvas with scroll bar
        self.addi_img_padding = 20
        _, _, meta_height, meta_width = testing_vars.meta_img.shape
        self.frame_height = int(meta_height / self.fov_rescale) + self.addi_img_padding * 2
        self.frame_width = int(meta_width / self.fov_rescale) + self.addi_img_padding * 2
        frame = tkinter.Frame(self.root, width=self.frame_width, height=self.frame_height, relief=tkinter.SUNKEN)
        frame.pack(expand=True, fill=tkinter.BOTH)
        # frame.grid_rowconfigure(0, weight=1)
        # frame.grid_columnconfigure(0, weight=1)
        hscroll = tkinter.Scrollbar(frame, orient=tkinter.HORIZONTAL)
        hscroll.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        # hscroll.grid(row=1, column=0, sticky=tkinter.E+tkinter.W)
        vscroll = tkinter.Scrollbar(frame)
        vscroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        # vscroll.grid(row=0, column=1, sticky=tkinter.N+tkinter.S)

        btn_height = 2 # 120
        canvas_width = min(self.root.winfo_screenwidth(), self.frame_width)
        canvas_height = min(self.root.winfo_screenheight()-btn_height*60, self.frame_height)
        print(" [*] Canvas size: ({}, {})".format(canvas_width, canvas_height))
        self.canvas = tkinter.Canvas(
            frame, width=canvas_width, height=canvas_height, bd=0, 
            xscrollcommand=hscroll.set, yscrollcommand=vscroll.set,
            scrollregion=(0, 0, self.frame_width, self.frame_height))
        self.canvas.pack(side=tkinter.LEFT, expand=True, fill=tkinter.BOTH)
        hscroll.config(command=self.canvas.xview)
        vscroll.config(command=self.canvas.yview)

        # Bind buttons
        # pixel = tkinter.PhotoImage(width=1, height=1)
        btn_common_args = {
            # "image": pixel,
            "height": btn_height
        }
        extra_btns = []
        btn_ssg = tkinter.Button(self.root, text="SS Global (all)", command=self.run_global_resample, **btn_common_args)
        btn_ssg.pack(side=tkinter.LEFT, padx=5)
        extra_btns.append(btn_ssg)
        if "FusedGenerationManager" in config.task.task_manager:
            for i in range(len(config.task.style_centers)):
                btn_ssg = tkinter.Button(
                    self.root, text="SS Global {}".format(i), **btn_common_args,
                    command=lambda index=i: self.run_global_resample(update_index=index)) # Dereference `i` with default arg
                btn_ssg.pack(side=tkinter.LEFT, padx=5)
                extra_btns.append(btn_ssg)
        btn_ssl = tkinter.Button(self.root, text="SS Local", command=self.run_local_resample, **btn_common_args)
        btn_ssl.pack(side=tkinter.LEFT, padx=5)
        btn_tsn = tkinter.Button(self.root, text="TS Noise", command=self.run_noises_resample, **btn_common_args)
        btn_tsn.pack(side=tkinter.LEFT, padx=5)
        btn_undo = tkinter.Button(self.root, text="Undo", command=self.run_undo, **btn_common_args)
        btn_undo.pack(side=tkinter.LEFT, padx=5)
        btn_redo = tkinter.Button(self.root, text="Redo", command=self.run_redo, **btn_common_args)
        btn_redo.pack(side=tkinter.LEFT, padx=5)
        btn_clean_bbox = tkinter.Button(self.root, text="Clean Bbox", command=self.clean_bbox, **btn_common_args)
        btn_clean_bbox.pack(side=tkinter.LEFT, padx=5)
        btn_comp_task = tkinter.Button(self.root, text="Save", command=self.dump_current, **btn_common_args)
        btn_comp_task.pack(side=tkinter.LEFT, padx=55)
        btn_comp_task = tkinter.Button(self.root, text="Complete & Save", command=self.set_task_complete, **btn_common_args)
        btn_comp_task.pack(side=tkinter.LEFT, padx=15)
        self.all_btns = [btn_ssg, btn_ssl, btn_tsn, btn_comp_task, btn_undo, btn_redo, btn_clean_bbox] + extra_btns


        self.pbar = tkinter.ttk.Progressbar(self.root, orient=tkinter.HORIZONTAL, length=200, mode='determinate')
        self.pbar.pack(side=tkinter.RIGHT, padx=5)
        self.pbar_caller = TkinterProgressBarWrapper(self.root, self.pbar)

        # Bind mouse click event
        self.canvas.bind("<Button 1>", self.mouse_left_click)
        self.canvas.bind("<Button 3>", self.mouse_right_click)
        self.canvas.config(scrollregion=self.canvas.bbox(tkinter.ALL))

        # Setup state variables
        self.latents_stack = []
        self.reset_states()

        # Run the first inference
        self.snapshot_cursor = 0 # Definition: The destination of next snapshot to place
        self.task_manager.generate(testing_vars)
        self.update_img(testing_vars, first_run=True)
        self.push_testing_vars_snapshot(testing_vars)

        # Launch
        self.root.mainloop()

    def reset_states(self):
        self.is_drawing_pri_bbox = False
        self.is_drawing_sec_bbox = False

        self.pri_bbox_stack = []
        self.sec_bbox_stack = []
        self.cur_bbox_entry = (0, 0)

        self.is_task_complete = False
        self.is_action_complete = False

    def update_img(self, testing_vars, with_bbox=False, first_run=False):
        assert testing_vars.meta_img.shape[0] == 1
        if with_bbox:
            bbox_stoke = 2 // 2
            meta_img = testing_vars.meta_img[0].clone()
            for bbox in self.pri_bbox_stack:
                bbox = self.calibrate_bbox_idx_order(bbox)
                xmax = meta_img.shape[1] - 1
                ymax = meta_img.shape[2] - 1
                meta_img[0, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = 1
                meta_img[1, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = -1
                meta_img[2, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = -1
                meta_img[0, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = 1
                meta_img[1, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = -1
                meta_img[2, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = -1
                meta_img[0, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = 1
                meta_img[1, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[2, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[0, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = 1
                meta_img[1, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[2, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
            for bbox in self.sec_bbox_stack:
                bbox = self.calibrate_bbox_idx_order(bbox)
                xmax = meta_img.shape[1] - 1
                ymax = meta_img.shape[2] - 1
                meta_img[0, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = -1
                meta_img[1, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = -1
                meta_img[2, bbox[0]:bbox[1], max(0, bbox[2]-bbox_stoke):min(bbox[2]+bbox_stoke, ymax)] = 1
                meta_img[0, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = -1
                meta_img[1, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = -1
                meta_img[2, bbox[0]:bbox[1], max(0, bbox[3]-bbox_stoke):min(bbox[3]+bbox_stoke, ymax)] = 1
                meta_img[0, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[1, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[2, max(0, bbox[0]-bbox_stoke):min(bbox[0]+bbox_stoke, xmax), bbox[2]:bbox[3]] = 1
                meta_img[0, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[1, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = -1
                meta_img[2, max(0, bbox[1]-bbox_stoke):min(bbox[1]+bbox_stoke, xmax), bbox[2]:bbox[3]] = 1
        else:
            meta_img = testing_vars.meta_img[0]
        image = self.cvt_tensor2pil(meta_img)
        image = image.resize(
           (int(image.width / self.fov_rescale), int(image.height / self.fov_rescale)), Image.LANCZOS)
        if self.addi_img_padding > 0:
            image = ImageOps.expand(image, self.addi_img_padding)

        # # Avoid GC collect...
        # https://stackoverflow.com/questions/16424091/why-does-tkinter-image-not-show-up-if-created-in-a-function
        self._image = ImageTk.PhotoImage(image=image)
        if first_run:
            self.canvas_image = self.canvas.create_image(0, 0, image=self._image, anchor="nw")
        else:
            self.canvas.itemconfig(self.canvas_image, image=self._image)

            
    ##########################################################
    ## Events
    ##########################################################

    def run_global_resample(self, update_index=None):
        self.disable_all_btns()
        testing_vars_snapshot = self.get_cur_testing_vars().clone()
        testing_vars_snapshot.update_global_latent(
            self.latent_sampler, self.task_manager.g_ema, mixing=False, update_index=update_index)
        testing_vars_snapshot.maybe_reset_to_inv_records(self.g_ema_module)
        self.task_manager.generate(
            testing_vars_snapshot, 
            tkinter_pbar=self.pbar_caller)
        self.update_img(testing_vars_snapshot)
        self.push_testing_vars_snapshot(testing_vars_snapshot)
        self.reset_states()
        self.enable_all_btns()

    def run_local_resample(self):
        if len(self.pri_bbox_stack) == 0: return
        self.disable_all_btns()
        testing_vars_snapshot = self.get_cur_testing_vars().clone()
        pri_selection_map = self.create_selection_map(target="local_latent", which="pri")
        sec_selection_map = self.create_selection_map(target="local_latent", which="sec")
        testing_vars_snapshot.update_local_latent(
            self.latent_sampler, pri_selection_map, ref_selection_map=sec_selection_map)
        testing_vars_snapshot.maybe_reset_to_inv_records(self.g_ema_module)
        self.task_manager.generate(
            testing_vars_snapshot, 
            tkinter_pbar=self.pbar_caller,
            update_by_ss_map=pri_selection_map)
        self.update_img(testing_vars_snapshot)
        self.push_testing_vars_snapshot(testing_vars_snapshot)
        self.reset_states()
        self.enable_all_btns()

    def run_noises_resample(self):
        if len(self.pri_bbox_stack) == 0: return
        self.disable_all_btns()
        testing_vars_snapshot = self.get_cur_testing_vars().clone()
        selection_maps = self.create_selection_map(target="noise")
        testing_vars_snapshot.update_noises(selection_maps)
        testing_vars_snapshot.maybe_reset_to_inv_records(self.g_ema_module)
        self.task_manager.generate(
            testing_vars_snapshot, 
            tkinter_pbar=self.pbar_caller,
            update_by_ts_map=selection_maps[0])
        self.update_img(testing_vars_snapshot)
        self.push_testing_vars_snapshot(testing_vars_snapshot)
        self.reset_states()
        self.enable_all_btns()

    def dump_current(self):
        cur_testing_vars = self.get_cur_testing_vars()
        self.task_manager.save_results(
            cur_testing_vars.meta_img, dump_vars=cur_testing_vars)

    def set_task_complete(self):
        self.is_task_complete = True
        self.dump_current()
        self.root.destroy()

    def run_undo(self):
        if self.snapshot_cursor > 1:
            self.snapshot_cursor -= 1
            print(" [ActionStack] After undo cursor {} ; len {}".format(self.snapshot_cursor, len(self.latents_stack)))
            self.update_img(self.get_cur_testing_vars())
            self.reset_states()
        else:
            print(" [*] No more undo...")

    def run_redo(self):
        if self.snapshot_cursor < len(self.latents_stack):
            self.snapshot_cursor += 1
            print(" [ActionStack] After redo cursor {} ; len {}".format(self.snapshot_cursor, len(self.latents_stack)))
            self.update_img(self.get_cur_testing_vars())
            self.reset_states()
        else:
            print(" [*] No more redo...")

    def clean_bbox(self):
        self.pri_bbox_stack = []
        self.sec_bbox_stack = []
        self.update_img(self.get_cur_testing_vars(), with_bbox=True)
        self.is_drawing_pri_bbox = False
        self.is_drawing_sec_bbox = False

    def mouse_left_click(self, event):
        rel_x = event.x
        rel_y = event.y
        x_view_st, _ = self.canvas.xview()
        y_view_st, _ = self.canvas.yview()
        abs_x = int(np.round(x_view_st * self.frame_width + rel_x))
        abs_y = int(np.round(y_view_st * self.frame_height + rel_y))

        # Discount padding
        abs_x = max(0, abs_x - self.addi_img_padding)
        abs_y = max(0, abs_y - self.addi_img_padding)

        # Discount view rescale
        abs_x = int(abs_x * self.fov_rescale)
        abs_y = int(abs_y * self.fov_rescale)

        print(" [Mouse left] X: rel ({}, {}), view ({:.4f}, {:.4f}), viewT ({:.4f}, {:.4f}), abs ({}, {})".format(
            rel_x, rel_y, 
            x_view_st, y_view_st,
            x_view_st*self.frame_width, y_view_st*self.frame_height, 
            abs_x, abs_y))

        if self.is_drawing_pri_bbox:
            cur_bbox_ending = (abs_y, abs_x)
            self.pri_bbox_stack.append([
                self.cur_bbox_entry[0], 
                cur_bbox_ending[0], 
                self.cur_bbox_entry[1], 
                cur_bbox_ending[1],
            ])
            self.update_img(self.get_cur_testing_vars(), with_bbox=True)
            self.is_drawing_pri_bbox = False
        else:
            self.cur_bbox_entry = (abs_y, abs_x)
            self.is_drawing_pri_bbox = True


    def mouse_right_click(self, event):
        rel_x = event.x
        rel_y = event.y
        x_view_st, _ = self.canvas.xview()
        y_view_st, _ = self.canvas.yview()
        abs_x = int(np.round(x_view_st * self.frame_width + rel_x))
        abs_y = int(np.round(y_view_st * self.frame_height + rel_y))

        # Discount padding
        abs_x = max(0, abs_x - self.addi_img_padding)
        abs_y = max(0, abs_y - self.addi_img_padding)

        # Discount view rescale
        abs_x = int(abs_x * self.fov_rescale)
        abs_y = int(abs_y * self.fov_rescale)

        print(" [Mouse right] X: rel ({}, {}), view ({:.4f}, {:.4f}), viewT ({:.4f}, {:.4f}), abs ({}, {})".format(
            rel_x, rel_y, 
            x_view_st, y_view_st,
            x_view_st*self.frame_width, y_view_st*self.frame_height, 
            abs_x, abs_y))

        if self.is_drawing_sec_bbox:
            cur_bbox_ending = (abs_y, abs_x)
            self.sec_bbox_stack.append([
                self.cur_bbox_entry[0], 
                cur_bbox_ending[0], 
                self.cur_bbox_entry[1], 
                cur_bbox_ending[1],
            ])
            self.update_img(self.get_cur_testing_vars(), with_bbox=True)
            self.is_drawing_sec_bbox = False
        else:
            self.cur_bbox_entry = (abs_y, abs_x)
            self.is_drawing_sec_bbox = True


    ##########################################################
    ## Utils funcs
    ##########################################################

    def has_pos_pix(self, tensor):
        return (tensor > 1e-6).any()

    def calibrate_bbox_idx_order(self, bbox):
        x1, x2, y1, y2 = bbox
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

    def binarize_selection_map(self, selection_map):
        selection_map_bin = torch.zeros_like(selection_map, dtype=torch.bool)
        selection_map_bin[selection_map > 1e-6] = 1
        return selection_map_bin.type(torch.float32)

    def create_selection_map(self, target=None, which=None):
        if target == "local_latent":
            assert which is not None, "Must specify pri or sec!"
        assert target in {"local_latent", "noise"}
        cur_testing_vars = self.get_cur_testing_vars()
        _, _, H, W = cur_testing_vars.meta_img.shape

        if target == "noise":
            device = cur_testing_vars.noises[0].device
        elif target == "local_latent":
            device = cur_testing_vars.local_latent.device
        else:
            raise NotImplementedError()

        if which == "sec":
            cur_bbox_stack = self.sec_bbox_stack
            if len(cur_bbox_stack)==0:
                return None
        else:
            cur_bbox_stack = self.pri_bbox_stack
        selection_map = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)
        for bbox in cur_bbox_stack:
            bbox = self.calibrate_bbox_idx_order(bbox)
            selection_map[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

        inv_selection_maps_ss, inv_selection_maps_ts, _, _ = \
            self.g_ema_module.calibrate_spatial_shape(
                selection_map, direction="backward", padding_mode="replicate", verbose=False)

        if target == "local_latent":
            return self.binarize_selection_map(inv_selection_maps_ss[0])
        elif target == "noise":
            # Noise uses output side
            inter_selection_maps = inv_selection_maps_ts[1:] + [selection_map]
            return [self.binarize_selection_map(m) for m in inter_selection_maps]
        else:
            raise NotImplementedError()

    def disable_all_btns(self):
        for btn in self.all_btns:
            btn["state"] = "disabled"
    
    def enable_all_btns(self):
        for btn in self.all_btns:
            btn["state"] = "normal"

    def cvt_tensor2pil(self, tensor):
        np_arr = ((tensor.clamp(-1, 1) + 1) * 127.5).numpy().transpose(1, 2, 0)
        np_arr = np_arr.astype(np.uint8)
        return Image.fromarray(np_arr)

    def get_cur_testing_vars(self):
        return self.latents_stack[self.snapshot_cursor-1]

    def push_testing_vars_snapshot(self, snapshot):
        if len(self.latents_stack) > self.snapshot_cursor: # Performed undo before, discard redo stack
            self.latents_stack = self.latents_stack[:self.snapshot_cursor]
        self.latents_stack.append(snapshot)
        self.snapshot_cursor += 1
        assert self.snapshot_cursor == len(self.latents_stack)
        print(" [ActionStack] After snapshot cursor {} ; len {}".format(self.snapshot_cursor, len(self.latents_stack)))
        # print(" [!] Stack {}, cursor {}".format(len(self.latents_stack), self.snapshot_cursor))
        if len(self.latents_stack) > 100:
            self.latents_stack = self.latents_stack[-100:]
            self.snapshot_cursor = 100

    
    

# if __name__ == "__main__":
#     root = Tk()

#     #setting up a tkinter canvas with scrollbars
#     frame = Frame(root, bd=2, relief=SUNKEN)
#     frame.grid_rowconfigure(0, weight=1)
#     frame.grid_columnconfigure(0, weight=1)
#     xscroll = Scrollbar(frame, orient=HORIZONTAL)
#     xscroll.grid(row=1, column=0, sticky=E+W)
#     yscroll = Scrollbar(frame)
#     yscroll.grid(row=0, column=1, sticky=N+S)
#     canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
#     canvas.grid(row=0, column=0, sticky=N+S+E+W)
#     xscroll.config(command=canvas.xview)
#     yscroll.config(command=canvas.yview)
#     frame.pack(fill=BOTH,expand=1)

#     #adding the image
#     File = askopenfilename(parent=root, initialdir="C:/", title='Choose an image.')
#     img = ImageTk.PhotoImage(file=File)
#     canvas.create_image(0, 0, image=img, anchor="nw")
#     canvas.config(scrollregion=canvas.bbox(ALL))

#     #function to be called when mouse is clicked
#     def printcoords(event):
#         #outputting x and y coords to console
#         print (event.x,event.y)
#     #mouseclick event
#     canvas.bind("<Button 1>",printcoords)

#     root.mainloop()