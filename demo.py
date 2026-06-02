import gradio as gr
import Nifty.method as _nifty_method
from Nifty.method import *
#from Nifty.networks import *

# For file management
import time
import os

# For displaying the novelty map and other stuff
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

import warnings; warnings.filterwarnings('ignore')

# Functions
def Nifty_gradio_interface_complete(im1,im2=None,rs=1.,T=100,k=10,patchsize=16,stride=4,width=256,height=256,octaves=4,renoise=.5,warmup=0,memory=True,seed=None,noise=None,spotsize=1/4,blend=False,blend_alpha=0.5,save=True,blend_map=None, manual_noise_seed:int=0):
    torch.manual_seed(manual_noise_seed)
    img_1 = Tensor_load(im1)
    img_2 = Tensor_load(im2) if im2 is not None else None
    size = (height, width)
    synth= Nifty(img_1,img_2,rs,T,k,patchsize,stride,size,octaves,renoise,warmup,False,memory,seed,None if noise == 0 else noise,spotsize,blend,blend_alpha,save, torch.tensor([[1,0]]).unsqueeze(0).unsqueeze(0).float().to(_nifty_method.device) if blend_map else None)
    
    if save:
        out_path = save_img_to_path(f"./results/demo/synth_{os.path.basename(im1)}_{round(time.time())}.png",synth)
    
    img_synth = get_synthesized_image(synth)
    
    if display_debug_mode:
        
        # Calculate Novelty Map
        P_exmpl = Patch_extraction(img_1, patchsize=patchsize, stride=1)
        P_synth = Patch_extraction(synth, patchsize=patchsize, stride=1)
        
        dist_map, novel_areas = get_nn_distance_map_and_novel_areas(synth, P_exmpl, P_synth, patchsize, height, width)
        
        return (img_synth, novel_areas)

    return (img_synth, img_synth) # Return the synthesized image on both sides of the ImageSlider when not in debug mode, to keep the layout consistent with the debug mode (where we display the novelty map on the right side)

def get_synthesized_image(synth):
    return (synth[0].permute(1, 2, 0).cpu().numpy() * .5 + .5).clip(0, 1).astype(np.float32)

def get_nn_distance_map_and_novel_areas(synth, P_exmpl, P_synth, patchsize, H, W):
    N = P_synth.size(2)
    ld = []
    lind = []
    for i in tqdm(range(N//5000+1)): # chunk by chunk for NN search, for memory

        X = P_exmpl[:,:,:] 
        X = X.squeeze(0)
        X2 = (X**2).sum(0).unsqueeze(0) 
        Y = P_synth[:,:,5000*i:5000*(i+1)].squeeze(0)
        Y2 = (Y**2).sum(0).unsqueeze(0) 
        D = Y2.transpose(1,0) - 2 * torch.matmul(Y.transpose(1,0),X) + X2 
        Dnn,indnn=torch.min(D,dim=1) # indnn: index of nearest ref patch for each synth patch, and Dnn the distance
        ld.append(Dnn.cpu())
        lind.append(indnn.cpu())

    Dnn=torch.cat(ld)
    indnn=torch.cat(lind)

    P_dist=P_synth*0
    P_dist=P_dist.view(3,patchsize,patchsize,P_dist.shape[-1])[:1]*0.
    P_dist[0,patchsize//2,patchsize//2,:]=Dnn
    dist=nn.Fold((H,W), patchsize, dilation=1, padding=0, stride=1)(P_dist.view(patchsize**2,-1))

    # Apply 'hot' colormap
    dist_norm = dist.cpu()[0].numpy()
    dist_norm = dist_norm / dist_norm.max()
    dist_map = plt.cm.hot(dist_norm)[:, :, :3].astype(np.float32)

    # compute a mask
    cutoff=0.2
    mask=(dist/dist.max()/cutoff)
    mask[dist/dist.max()>cutoff]=1
    mask=.8*mask+.2 #keep backround dimmer

    novelty=synth*mask

    novel_areas = (novelty[0].permute(1, 2, 0).cpu().numpy() * .5 + .5).clip(0, 1).astype(np.float32)

    return dist_map, novel_areas

def save_img_to_path(file_path, synthetized_image)->str:    
    plt.imsave(file_path, get_synthesized_image(synthetized_image))
    return file_path

def update_processing_unit_selection(choice:str):
    _nifty_method.device = manually_select_device(try_gpu=(choice == "GPU"))

def resize_input_image(image_path:str, width:int, height:int)->str:
    img = Image.open(image_path)
    img_resized = img.resize((width, height))
    os.makedirs("./results/demo/resized/", exist_ok=True)
    resized_image_path = f"./results/demo/resized/{os.path.basename(image_path)}_resized.png"
    img_resized.save(resized_image_path)
    return resized_image_path

def update_output_display_mode(debug_mode_enabled:bool):
    global display_debug_mode
    display_debug_mode = debug_mode_enabled
# CSS
custom_css = """
.full_height {height: -webkit-fill-available !important;}
.full_width {width: -webkit-fill-available !important;}
.filled_flex_display {display: flex !important; align-content: stretch; justify-content: space-between;}
.filled_flex_display > div{display: grid !important; align-content: stretch; align-items: stretch; justify-items: stretch; flex-grow: 1 !important;}



#title{
    font-size: 48px; 
    font-weight: 700; 
    margin: 0; 
    color: #ffffff; 
    letter-spacing: -0.02em;
}
#subtitle{
    font-size: 20px; 
    color: #86868b; 
    margin:0;
    font-weight: 400;
}
#authors{
    font-size: 15px; 
    color: #86868b; 
    margin:0;
    font-style: italic;
}
.radio_group .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
}
footer {visibility: hidden}
"""

# Head
custom_head = """
<title>Nifty</title>
<meta name="Nifty demo app" content="A non-local image flow matching for texture synthesis">
"""

# Initialize processing unit selection
selected_processing_unit_type = "GPU" if "cuda" in str(_nifty_method.device) else "CPU"

display_debug_mode = False

# Interface
with gr.Blocks(title="Nifty") as nifty_demo:
    # Header and logo
    gr.HTML("""
            <header><a href="https://www.greyc.fr/"><img src="https://greycflix.greyc.fr/demo-portal/images/logo-GREYC-dark.svg" style="position: absolute; width: 12em;"></a></header>
    """)
    gr.HTML("""
        <div style="text-align: center; padding: 0px;">
            <h1 id="title">
                NIFTY
            </h1>
            <p id="subtitle">
                A NON-LOCAL IMAGE FLOW MATCHING FOR TEXTURE SYNTHESIS
            </p>
            <p id="authors">Pierrick Chatillon, Julien Rabin, David Tschumperlé</p>
        </div>
    """)
    
    # CPU or GPU selection
    with gr.Row():
        with gr.Column(scale=1):
            in_processing_unit_choice = gr.Radio(
                label="Mode",
                choices=["GPU", "CPU"],
                value=selected_processing_unit_type,
                type="value",
                elem_classes="radio_group",
                elem_id="processing_unit_radio_group"
            )
            in_debug_mode = gr.Checkbox(label="Debug Mode", value=False, info="Enable debug mode to display novelty areas on the output, which can help to understand and analyze the synthesis process, but can be slower and more memory consuming")
    
    # Inputs and outputs
    with gr.Row():
        # Input column
        with gr.Column(scale=1, elem_classes="full_height"):
            with gr.Row(scale=1, elem_classes="full_height"):
                with gr.Column(scale=1, elem_classes="full_height filled_flex_display"):
                    in_compressed_height = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Height",
                        info="Input image height in pixels once resized"
                    )
                    in_compressed_width = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Width",
                        info="Input image width in pixels once resized"
                    )
                    in_resize_button = gr.Button("Resize", variant="secondary")
                with gr.Column(scale=1, elem_classes="full_height"):
                    in_img1 = gr.Image(
                        label="Input Image",
                        value="results/red_peppers.jpg",
                        type="filepath",
                        width=256,
                        height=256,
                        elem_classes="full_height full_width"
                    ) 
                # Output column
        
        with gr.Column(scale=1, elem_classes="full_height"):
            with gr.Row(scale=1, elem_classes="full_height"):
                # The ImageSlider displays the synthesized image on the left (or not if not in debug mode) and the novel areas on the right
                with gr.Column(scale=1, elem_classes="full_height"):
                    out_img = gr.ImageSlider(
                        label="Output",
                        elem_id="output_image_slider",
                        elem_classes="full_height full_width"
                    )
                with gr.Column(scale=1, elem_classes="full_height filled_flex_display"):
                    in_height = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Height",
                        info="Output image height in pixels"
                    )
                    in_width = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Width",
                        info="Output image width in pixels"
                    )     
                    submit_btn = gr.Button("Generate", variant="primary")
        

    
    with gr.Row():
        in_rs = gr.Slider(
                minimum=0.,
                maximum=1.,
                value=1., 
                step=0.01,
                label="Ratio Sample",
                info="Ratio of the reference patches to sample at each step"
                )
        in_T = gr.Slider(
            minimum=0.,
            maximum=100.,
            value=50., # A vérifier, c'était marqué 50 (mais marqué entre 0 et 1)
            step=0.,
            label="Discretization steps",
            info="Number of (linear) distretization steps between 0 and 1 to solve the flow ODE"
            )
        in_k = gr.Slider(
            minimum=0.,
            maximum=50.,
            value=5.,
            step=0.,
            label="K Neighbors",
            info="Number of top closest patch used to approximate the velocity field"
            )
        in_octaves = gr.Slider(
            minimum=0.,
            maximum=20.,
            value=4.,
            step=1.,
            label="Octaves",
            info="Number of diadic scales used for the synthesis"
            )
        in_renoise = gr.Slider(
            minimum=0.,
            maximum=1.,
            value=0.3,
            step=0.01,
            label="Renoise",
            info="Time used renoise the smooth upsampled image at each resolution"
            )
    
    
    # Blending inputs
    with gr.Accordion("Blending Inputs", open=False):
                in_img2 = gr.Image(label="Blend Image (optional)",type="filepath") 
                in_blend = gr.Checkbox(value=False, label="Blend", info="Blend the synthesized image with the input image, which can help to preserve some of the structure of the input image")
                in_blend_alpha = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Blend Alpha", info="Alpha used for blending the synthesized image with the input image, if blending is enabled")
                in_blend_map = gr.Checkbox(label="Blend Map", info="Use a blending map for blending the two images (if not enabled, the blending is uniform across the image). For the demo it is set to : `torch.tensor([[1,0]]).unsqueeze(0).unsqueeze(0).float().to(device)`")
    
    
    # Optional/Advanced inputs
    with gr.Accordion("Advanced Inputs", open=False):
                
                in_patch_size = gr.Slider(minimum=1, maximum=50, value=16, step=1, label="Patch Size")
                in_stride = gr.Slider(minimum=0, maximum=999999, value=4, step=1, label="Stride")
                
                in_warmup = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Warmup", info="Number of initial steps during which the flow is not applied, which can help to stabilize the synthesis at the beginning")
                in_memory = gr.Checkbox(value=False,label="Memory", info="Use the memory efficient version of Nifty, which does not store all the intermediate synthesized images during the flow integration, but only the current one")
                in_seed = gr.Slider(minimum=0, maximum=999999, value=0, step=1, label="Seed")
                in_noise = gr.Checkbox(value = None,label="Noise", info="Add noise during the synthesis, which can help to escape local minima and produce more diverse results")
                in_spot_size = gr.Slider(minimum=1/4, maximum=1, value=0.25, step=0.01, label="Spot Size", info="Size of the spots used for the synthesis, as a ratio of the patch size")
                in_save = gr.Checkbox(value=False,label="Save", info="Save the synthesized image at the end of the synthesis")
    
    

    in_resize_button.click(
        fn = resize_input_image,
        inputs = [in_img1, in_compressed_width, in_compressed_height],
        outputs = in_img1
    )

    # Bind the click event to the function
    submit_btn.click(
        fn=Nifty_gradio_interface_complete,
        inputs = [in_img1,in_img2,in_rs,in_T,in_k,in_patch_size,in_stride,in_width,in_height,in_octaves,in_renoise,in_warmup,in_memory,in_seed,in_noise,in_spot_size,in_blend,in_blend_alpha,in_save,in_blend_map, in_seed],
        outputs = [out_img],
    )

    in_processing_unit_choice.input(
          fn = update_processing_unit_selection,
          inputs=in_processing_unit_choice,
          outputs=[]
    )
    in_debug_mode.change(
         fn=update_output_display_mode,
         inputs=in_debug_mode,
    )
    gr.Examples(
         label="Examples (Click to load the parameters) - Published in the paper and more",
          examples=[
                ["results/red_peppers.jpg",None,1.,100,10,16,4,512,256,4,0.5,0,True,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use
                ["comparison/eval_base/7.png","comparison/eval_base/8.png",1.,50,10,16,4,256,256,3,0.5,0,True,None,None,1/4,True,0.5,False,False,0], # Pixel-level blending
                ["comparison/eval_base/7.png","comparison/eval_base/8.png",1.,50,10,16,4,256,256,3,0.5,0,True,None,None,1/4,False,0.5,False,False,0], # Distribution-level blending
                ["comparison/eval_base/7.png","comparison/eval_base/8.png",1.,50,10,16,4,256*3,256,3,0.5,0,True,None,None,1/4,True,0.5,False,True,0], # Spatial interpolation
                ["comparison/eval_base/1.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #1
                ["comparison/eval_base/2.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #2
                ["comparison/eval_base/3.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #3
                ["comparison/eval_base/4.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #4
                ["comparison/eval_base/5.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #5
                ["comparison/eval_base/6.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #6
                ["comparison/eval_base/7.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #7
                ["comparison/eval_base/8.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #8
                ["comparison/eval_base/9.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #9
                ["comparison/eval_base/10.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #10
                ["comparison/eval_base/11.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #11
                ["comparison/eval_base/12.png",None,1.,100,10,16,4,512,512,4,0.5,0,False,None,None,1/4,False,0.5,False,False,0], # Basic texture synthesis use, Paper #12
          ],
          inputs=[in_img1,in_img2,in_rs,in_T,in_k,in_patch_size,in_stride,in_width,in_height,in_octaves,in_renoise,in_warmup,in_memory,in_seed,in_noise,in_spot_size,in_blend,in_blend_alpha,in_save,in_blend_map, in_seed],
          )
    
    # Footer and links to the paper and code
    gr.HTML("""
        <div style="text-align: center; padding: 0px;;margin-top:30px;">
            <a href="https://hal.science/hal-05287967">HAL</a>
            <a href="https://github.com/PierrickCh/Nifty">Github</a>
            <a href="https://arxiv.org/abs/2509.22318" >ArXiv</a>
        </div>
    """)
# Run the Nifty demo
nifty_demo.launch(head=custom_head,share=False, css=custom_css)