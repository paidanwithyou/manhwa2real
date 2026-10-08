"""
chiroai.py - ChiroAI: AI Manhwa Author, Editor & Alight Motion Designer
GitHub Ready | Single File
Cara kerja: AI buat cerita manhwa -> auto generate panel -> AI editor styling -> export motion graphics

Author: ChiroAI Community
Repo: Push file ini ke GitHub, tinggal run
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import json
from datetime import datetime
import numpy as np

# --- Load Models ---
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2TokenizerFast
from diffusers import AutoPipelineForImage2Image, StableDiffusionPipeline
import anthropic  # ChiroAI buat story generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🎨 ChiroAI Starting... Device: {DEVICE}")

# ============================================================
# PART 1: MANHWA STORY GENERATION (ChiroAI Author)
# ============================================================

client = anthropic.Anthropic()  # ChiroAI buat nulis cerita

STORY_SYSTEM_PROMPT = """You are ChiroAI - A Professional Manhwa Story Writer & Author.
Your expertise:
- Creating compelling Korean manhwa storylines
- Developing character arcs and dialogue
- Structuring scenes for visual storytelling
- Writing realistic Korean dialogue and expressions

Output format MUST be JSON with this structure:
{
    "title": "Manhwa Title",
    "episode": 1,
    "panels": [
        {
            "panel_num": 1,
            "character": "Character Name",
            "expression": "happy/sad/angry/neutral/shocked",
            "scene_description": "Visual description for AI image generation",
            "dialogue": "Character speech",
            "panel_type": "full_page/half/quarter",
            "background": "indoor/outdoor/school/night"
        }
    ],
    "summary": "Episode summary"
}"""

def generate_manhwa_story(concept, episode_num=1, panel_count=6):
    """AI Author: Generate manhwa story & panels"""
    
    prompt = f"""Write a Korean manhwa story based on this concept: {concept}
    
    Episode {episode_num}:
    - Create {panel_count} panels with clear visual descriptions
    - Include character names, expressions, and dialogue
    - Describe backgrounds and scene composition
    - Make dialogue sound natural and Korean
    
    Return as valid JSON only."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=STORY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        story_json = json.loads(message.content[0].text)
        return story_json
    except json.JSONDecodeError:
        return {"error": "Story generation failed", "raw": message.content[0].text}

# ============================================================
# PART 2: PANEL IMAGE GENERATION (AI Illustrator)
# ============================================================

# Text-to-Image untuk panel manhwa
txt2img_pipe = StableDiffusionPipeline.from_pretrained(
    "DreamShaper/DreamShaperXL",
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

if DEVICE == "cuda":
    txt2img_pipe.enable_attention_slicing()

# Image-to-Image untuk styling
img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=DTYPE,
    use_safetensors=True
).to(DEVICE)

# BLIP untuk baca existing images
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large", torch_dtype=DTYPE
).to(DEVICE)

MANHWA_PROMPT_STYLE = "high quality manga panel, detailed illustration, korean manhwa style, professional comic art, clear lines, dramatic shadows, expressive eyes, 4k"
NEGATIVE_PROMPT = "blurry, low quality, watermark, deformed, extra fingers, bad anatomy"

def generate_panel_image(scene_desc, character, expression, background):
    """AI Illustrator: Generate panel image dari description"""
    
    # Build prompt dengan info lengkap
    prompt = f"({MANHWA_PROMPT_STYLE}), {character} with {expression} expression, {scene_desc}, {background} background, manga art style"
    
    print(f"🎨 Generating panel: {prompt[:60]}...")
    
    image = txt2img_pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=512,
        width=512,
        num_inference_steps=20,
    ).images[0]
    
    return image

def apply_panel_style(image, style_type="noir", enhancement=0.7):
    """AI Editor: Enhance panel dengan styling"""
    
    style_prompts = {
        "noir": "dark noir style, high contrast, dramatic lighting, black and white tones",
        "dramatic": "dramatic lighting, intense emotions, vivid colors, cinematic",
        "realistic": "photorealistic, detailed textures, natural lighting, korean drama style",
        "bright": "bright colors, optimistic, clear lighting, school setting vibes"
    }
    
    style_prompt = style_prompts.get(style_type, style_prompts["dramatic"])
    
    result = img2img_pipe(
        prompt=f"manhwa panel, {style_prompt}",
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=enhancement,
        guidance_scale=7.5,
        num_inference_steps=15,
    ).images[0]
    
    return result

# ============================================================
# PART 3: PANEL LAYOUT & EDITING (AI Editor Alight Motion)
# ============================================================

def add_dialogue_bubble(image, text, bubble_type="round", position="bottom"):
    """AI Editor: Add dialogue bubble ke panel"""
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load font, fallback ke default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    width, height = img_copy.size
    
    # Bubble dimensions
    bubble_height = 80
    bubble_width = min(400, width - 40)
    
    if position == "bottom":
        bubble_y = height - bubble_height - 20
    elif position == "top":
        bubble_y = 20
    else:
        bubble_y = height // 2
    
    bubble_x = (width - bubble_width) // 2
    
    # Draw bubble (rounded rectangle)
    bubble_box = [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height]
    draw.rounded_rectangle(bubble_box, radius=15, fill="white", outline="black", width=2)
    
    # Draw tail (triangle shape)
    if position == "bottom":
        tail_y = bubble_y + bubble_height
        tail_points = [(bubble_x + 50, tail_y), (bubble_x + 80, tail_y + 20), (bubble_x + 30, tail_y)]
        draw.polygon(tail_points, fill="white", outline="black")
    
    # Add text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = bubble_x + (bubble_width - text_width) // 2
    text_y = bubble_y + (bubble_height - text_height) // 2
    
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return img_copy

def add_character_name(image, name, position="top_left"):
    """AI Editor: Add character name label"""
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    width, height = img_copy.size
    
    # Position
    positions = {
        "top_left": (10, 10),
        "top_right": (width - 150, 10),
        "bottom_left": (10, height - 30),
        "bottom_right": (width - 150, height - 30)
    }
    
    pos = positions.get(position, positions["top_left"])
    
    # Draw background for text
    text_bbox = draw.textbbox(pos, name, font=font)
    draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
                   fill="black", outline="white", width=2)
    draw.text(pos, name, fill="white", font=font)
    
    return img_copy

def create_panel_layout(panels_data, layout_type="4panels"):
    """AI Editor: Create page layout dengan multiple panels"""
    
    if layout_type == "4panels":
        # 2x2 layout
        canvas = Image.new("RGB", (1024, 1024), "white")
        panel_size = (500, 500)
        positions = [(0, 0), (524, 0), (0, 524), (524, 524)]
    
    elif layout_type == "6panels":
        # 3x2 layout
        canvas = Image.new("RGB", (1024, 680), "white")
        panel_size = (340, 340)
        positions = [(0, 0), (342, 0), (684, 0), (0, 342), (342, 342), (684, 342)]
    
    else:  # full_page
        return panels_data[0] if panels_data else Image.new("RGB", (512, 512))
    
    # Paste panels
    for i, (panel, pos) in enumerate(zip(panels_data, positions)):
        if panel:
            resized = panel.resize(panel_size)
            canvas.paste(resized, pos)
    
    return canvas

# ============================================================
# PART 4: ALIGHT MOTION EXPORT (Animation Designer)
# ============================================================

def export_motion_data(panels_list, title, episode):
    """Export data untuk Alight Motion animation"""
    
    motion_project = {
        "project_name": f"{title} - Episode {episode}",
        "created_at": datetime.now().isoformat(),
        "timeline": {
            "duration_ms": len(panels_list) * 2000,  # 2 seconds per panel
            "fps": 30,
            "panels": []
        },
        "animations": []
    }
    
    # Add panel keyframes
    for idx, panel in enumerate(panels_list):
        keyframe = {
            "panel_id": idx,
            "start_time_ms": idx * 2000,
            "duration_ms": 2000,
            "transitions": [
                {
                    "type": "fade",
                    "duration_ms": 300,
                    "easing": "ease_in_out"
                },
                {
                    "type": "slide",
                    "direction": "left_to_right",
                    "duration_ms": 500
                }
            ]
        }
        motion_project["timeline"]["panels"].append(keyframe)
    
    # Add particle effects
    for idx in range(len(panels_list) - 1):
        motion_project["animations"].append({
            "type": "particle_burst",
            "panel": idx,
            "time_ms": (idx + 1) * 2000 - 300,
            "intensity": 0.7,
            "color": "#ff6b6b"
        })
    
    return motion_project

def save_motion_json(motion_data, filename="manhwa_motion.json"):
    """Save motion data untuk Alight Motion"""
    
    with open(filename, 'w') as f:
        json.dump(motion_data, f, indent=2)
    
    return filename

# ============================================================
# PART 5: MAIN WORKFLOW (Story -> Panels -> Layout -> Motion)
# ============================================================

def chiro_full_workflow(concept, panel_style="dramatic", layout="4panels", export_motion=True):
    """ChiroAI Full Workflow: Story -> Panels -> Layout -> Motion Export"""
    
    results = {
        "story": None,
        "panels": [],
        "layout": None,
        "motion_file": None,
        "metadata": {}
    }
    
    # Step 1: Generate Story
    print("📖 Step 1: ChiroAI Author generating story...")
    story = generate_manhwa_story(concept, panel_count=4)
    results["story"] = story
    
    if "error" in story:
        return results
    
    # Step 2: Generate Panel Images
    print("🎨 Step 2: AI Illustrator generating panels...")
    for panel in story.get("panels", [])[:4]:
        try:
            img = generate_panel_image(
                panel.get("scene_description", ""),
                panel.get("character", ""),
                panel.get("expression", "neutral"),
                panel.get("background", "indoor")
            )
            
            # Apply styling
            img = apply_panel_style(img, panel_style)
            
            # Add dialogue if exists
            if panel.get("dialogue"):
                img = add_dialogue_bubble(img, panel["dialogue"])
            
            # Add character name
            if panel.get("character"):
                img = add_character_name(img, panel["character"])
            
            results["panels"].append(img)
        except Exception as e:
            print(f"❌ Panel generation error: {e}")
            continue
    
    # Step 3: Create Layout
    print(f"📐 Step 3: AI Editor creating {layout} layout...")
    if results["panels"]:
        results["layout"] = create_panel_layout(results["panels"], layout)
    
    # Step 4: Export Motion Data
    if export_motion and results["panels"]:
        print("✨ Step 4: AI Animation Designer exporting motion data...")
        motion_data = export_motion_data(
            results["panels"],
            story.get("title", "Untitled"),
            story.get("episode", 1)
        )
        motion_file = save_motion_json(motion_data)
        results["motion_file"] = motion_file
        print(f"💾 Motion file saved: {motion_file}")
    
    results["metadata"] = {
        "title": story.get("title", "Untitled"),
        "episode": story.get("episode", 1),
        "panel_count": len(results["panels"]),
        "layout_type": layout,
        "style": panel_style
    }
    
    return results

# ============================================================
# PART 6: GRADIO UI (Dashboard)
# ============================================================

def process_workflow(concept, style, layout, include_motion):
    """Process full workflow dan return results"""
    
    results = chiro_full_workflow(
        concept,
        panel_style=style,
        layout=layout,
        export_motion=include_motion
    )
    
    # Return info
    info_text = f"""
    ✅ ChiroAI Workflow Complete!
    
    📖 Title: {results['metadata'].get('title', 'N/A')}
    🎬 Episode: {results['metadata'].get('episode', 'N/A')}
    🎨 Panels Generated: {results['metadata'].get('panel_count', 0)}
    📐 Layout: {layout}
    💾 Motion Export: {'✓' if results['motion_file'] else '✗'}
    """
    
    layout_image = results.get("layout")
    
    return layout_image, info_text

# Build UI
with gr.Blocks(title="ChiroAI - Manhwa Author & Editor") as demo:
    gr.Markdown("""
    # 🎨 ChiroAI - Professional Manhwa Creator
    **AI Author** → Story Writing | **AI Illustrator** → Panel Generation | **AI Editor** → Styling & Layout | **Animation Designer** → Alight Motion Export
    """)
    
    with gr.Tabs():
        # Tab 1: Full Workflow
        with gr.Tab("Full Workflow"):
            gr.Markdown("### Generate Manhwa dari konsep → selesai dengan motion graphics!")
            
            with gr.Row():
                with gr.Column():
                    concept_input = gr.Textbox(
                        label="Manhwa Concept",
                        placeholder="cth: A mysterious romance in Seoul high school...",
                        lines=3
                    )
                    
                    style_select = gr.Radio(
                        ["dramatic", "noir", "realistic", "bright"],
                        value="dramatic",
                        label="Panel Style"
                    )
                    
                    layout_select = gr.Radio(
                        ["4panels", "6panels", "full_page"],
                        value="4panels",
                        label="Page Layout"
                    )
                    
                    motion_toggle = gr.Checkbox(
                        value=True,
                        label="Export Alight Motion Data"
                    )
                    
                    process_btn = gr.Button("🚀 CREATE MANHWA", variant="primary", scale=2)
                
                with gr.Column():
                    layout_output = gr.Image(label="Generated Page Layout")
                    info_output = gr.Textbox(label="Workflow Info")
            
            process_btn.click(
                process_workflow,
                inputs=[concept_input, style_select, layout_select, motion_toggle],
                outputs=[layout_output, info_output]
            )
        
        # Tab 2: Story Generation Only
        with gr.Tab("Story Writer (AI Author)"):
            with gr.Row():
                with gr.Column():
                    story_concept = gr.Textbox(
                        label="Story Concept",
                        placeholder="Describe your manhwa idea...",
                        lines=4
                    )
                    episode_num = gr.Slider(1, 20, value=1, step=1, label="Episode Number")
                    panels_count = gr.Slider(4, 12, value=6, step=1, label="Panel Count")
                    story_btn = gr.Button("📖 Generate Story", variant="primary")
                
                with gr.Column():
                    story_output = gr.JSON(label="Generated Story")
            
            story_btn.click(
                lambda c, e, p: generate_manhwa_story(c, e, int(p)),
                inputs=[story_concept, episode_num, panels_count],
                outputs=[story_output]
            )
        
        # Tab 3: Panel Generation Only
        with gr.Tab("Panel Generator (AI Illustrator)"):
            with gr.Row():
                with gr.Column():
                    scene_desc = gr.Textbox(
                        label="Scene Description",
                        placeholder="A girl standing in the rain, looking sad...",
                        lines=3
                    )
                    character_name = gr.Textbox(label="Character Name", value="Girl")
                    character_expr = gr.Radio(
                        ["happy", "sad", "angry", "neutral", "shocked"],
                        value="sad",
                        label="Expression"
                    )
                    background_type = gr.Radio(
                        ["indoor", "outdoor", "school", "night", "café"],
                        value="outdoor",
                        label="Background"
                    )
                    panel_style = gr.Radio(
                        ["dramatic", "noir", "realistic", "bright"],
                        value="dramatic",
                        label="Style"
                    )
                    gen_panel_btn = gr.Button("🎨 Generate Panel", variant="primary")
                
                with gr.Column():
                    panel_output = gr.Image(label="Generated Panel")
            
            gen_panel_btn.click(
                generate_panel_image,
                inputs=[scene_desc, character_name, character_expr, background_type],
                outputs=[panel_output]
            )

gr.Markdown("""
### 🚀 ChiroAI Features:
- **📖 AI Author**: Professional story generation using Claude
- **🎨 AI Illustrator**: Panel creation with Stable Diffusion & RealVisXL
- **✏️ AI Editor**: Styling, dialogue bubbles, character labels
- **🎬 Animation Designer**: Export to Alight Motion JSON format
- **📐 Layout Engine**: Multiple panel configurations (4, 6 panels, full-page)

### 💡 Tips:
- Detailed scene descriptions = better results
- Style choices affect panel mood
- Motion export ready for Alight Motion Pro
- Export as high-quality PNG for print
""")

if __name__ == "__main__":
    demo.launch()
