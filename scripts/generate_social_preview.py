"""
Generate a professional, high-impact GitHub Social Preview Image (1280x640 px).
Uses 2x supersampling (2560x1280) downscaled with Lanczos for ultra-crisp anti-aliasing.
"""

import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def create_social_preview(output_path="assets/social-preview.png"):
    # Target size: 1280x640. Supersample at 2x: 2560x1280
    SCALE = 2
    W, H = 1280 * SCALE, 640 * SCALE

    # Create base canvas with dark modern gradient background
    canvas = Image.new("RGBA", (W, H), (11, 15, 25, 255))  # #0b0f19
    draw = ImageDraw.Draw(canvas)

    # 1. Background radial / gradient glows
    # Indigo glow top-left
    glow1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g1_draw = ImageDraw.Draw(glow1)
    g1_draw.ellipse(
        [int(-200 * SCALE), int(-200 * SCALE), int(800 * SCALE), int(700 * SCALE)],
        fill=(79, 70, 229, 60),
    )
    glow1 = glow1.filter(ImageFilter.GaussianBlur(160 * SCALE))
    canvas = Image.alpha_composite(canvas, glow1)

    # Teal / Cyan glow bottom-right
    glow2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g2_draw = ImageDraw.Draw(glow2)
    g2_draw.ellipse(
        [int(1500 * SCALE), int(200 * SCALE), int(2700 * SCALE), int(1400 * SCALE)],
        fill=(13, 148, 136, 50),
    )
    glow2 = glow2.filter(ImageFilter.GaussianBlur(180 * SCALE))
    canvas = Image.alpha_composite(canvas, glow2)

    # Re-obtain draw handle
    draw = ImageDraw.Draw(canvas)

    # Subtle grid overlay for tech aesthetic
    grid_color = (255, 255, 255, 6)
    for x in range(0, W, 80 * SCALE):
        draw.line([(x, 0), (x, H)], fill=grid_color, width=1 * SCALE)
    for y in range(0, H, 80 * SCALE):
        draw.line([(0, y), (W, y)], fill=grid_color, width=1 * SCALE)

    # Outer border
    draw.rounded_rectangle(
        [16 * SCALE, 16 * SCALE, W - 16 * SCALE, H - 16 * SCALE],
        radius=24 * SCALE,
        outline=(255, 255, 255, 24),
        width=2 * SCALE,
    )

    # Fonts
    font_bold_path = r"C:\Windows\Fonts\segoeuib.ttf"
    font_reg_path = r"C:\Windows\Fonts\segoeui.ttf"
    if not os.path.exists(font_bold_path):
        font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"
        font_reg_path = r"C:\Windows\Fonts\arial.ttf"

    font_badge = ImageFont.truetype(font_bold_path, 13 * SCALE)
    font_title = ImageFont.truetype(font_bold_path, 42 * SCALE)
    font_sub = ImageFont.truetype(font_reg_path, 18 * SCALE)
    ImageFont.truetype(font_bold_path, 14 * SCALE)
    font_body = ImageFont.truetype(font_reg_path, 15 * SCALE)
    font_body_bold = ImageFont.truetype(font_bold_path, 15 * SCALE)
    font_code = ImageFont.truetype(font_reg_path, 13 * SCALE)
    ImageFont.truetype(font_reg_path, 12 * SCALE)
    font_small_bold = ImageFont.truetype(font_bold_path, 12 * SCALE)

    # ----------------- LEFT PANEL: BRANDING & HIGHLIGHTS -----------------
    left_x = 64 * SCALE
    top_y = 60 * SCALE

    # Top Category Pill
    pill_text = "PRODUCTION-GRADE AGENTIC RAG"
    pill_w, pill_h = 280 * SCALE, 32 * SCALE
    draw.rounded_rectangle(
        [left_x, top_y, left_x + pill_w, top_y + pill_h],
        radius=16 * SCALE,
        fill=(30, 41, 59, 200),
        outline=(99, 102, 241, 140),
        width=1 * SCALE,
    )
    # Glow dot in pill
    draw.ellipse(
        [left_x + 12 * SCALE, top_y + 11 * SCALE, left_x + 22 * SCALE, top_y + 21 * SCALE],
        fill=(99, 102, 241, 255),
    )
    draw.text(
        (left_x + 30 * SCALE, top_y + 8 * SCALE), pill_text, fill=(224, 231, 255), font=font_badge
    )

    # Title: 2 lines
    y_title = top_y + 48 * SCALE
    draw.text((left_x, y_title), "Agentic Customer", fill=(255, 255, 255), font=font_title)
    draw.text((left_x, y_title + 50 * SCALE), "Support RAG", fill=(129, 140, 248), font=font_title)

    # Subtitle
    y_sub = y_title + 112 * SCALE
    draw.text(
        (left_x, y_sub),
        "Self-Correcting State Machine with Scope Gates,\nNeural Reranking & Claim Hallucination Defense",
        fill=(148, 163, 184),
        font=font_sub,
        spacing=6 * SCALE,
    )

    # Key Capabilities Box
    y_caps = y_sub + 72 * SCALE
    capabilities = [
        ("🛡️", "Pre-retrieval scope gating & prompt injection defense"),
        ("🎯", "Neural cross-encoder reranking & evidence grading"),
        ("✅", "Post-generation claim verification & structured citations"),
        ("🧰", "Tenant-scoped order lookup & automatic human escalation"),
    ]

    for icon, text in capabilities:
        draw.text((left_x, y_caps), icon, font=font_body)
        draw.text((left_x + 28 * SCALE, y_caps), text, fill=(203, 213, 225), font=font_body)
        y_caps += 30 * SCALE

    # Stack Badges Row at bottom left
    y_stack = H - 96 * SCALE
    tech_tags = [
        ("LangGraph", (99, 102, 241)),
        ("FastAPI", (16, 185, 129)),
        ("Pinecone Serverless", (236, 72, 153)),
        ("React 19", (14, 165, 233)),
        ("235 Tests Passing", (34, 197, 94)),
    ]

    tag_x = left_x
    for name, col in tech_tags:
        bbox = font_small_bold.getbbox(name)
        tw = bbox[2] - bbox[0]
        pad_x = 12 * SCALE
        tag_w = tw + pad_x * 2 + 12 * SCALE
        tag_h = 28 * SCALE
        draw.rounded_rectangle(
            [tag_x, y_stack, tag_x + tag_w, y_stack + tag_h],
            radius=14 * SCALE,
            fill=(15, 23, 42, 220),
            outline=(col[0], col[1], col[2], 120),
            width=1 * SCALE,
        )
        draw.ellipse(
            [tag_x + 10 * SCALE, y_stack + 9 * SCALE, tag_x + 18 * SCALE, y_stack + 17 * SCALE],
            fill=col,
        )
        draw.text(
            (tag_x + 24 * SCALE, y_stack + 6 * SCALE),
            name,
            fill=(241, 245, 249),
            font=font_small_bold,
        )
        tag_x += tag_w + 8 * SCALE

    # ----------------- RIGHT PANEL: INTERACTIVE CHAT UI MOCKUP -----------------
    card_x = 640 * SCALE
    card_y = 60 * SCALE
    card_w = 576 * SCALE
    card_h = 520 * SCALE

    # Card shadow
    shadow_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_img)
    s_draw.rounded_rectangle(
        [
            card_x - 4 * SCALE,
            card_y + 10 * SCALE,
            card_x + card_w + 4 * SCALE,
            card_y + card_h + 16 * SCALE,
        ],
        radius=20 * SCALE,
        fill=(0, 0, 0, 160),
    )
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(16 * SCALE))
    canvas = Image.alpha_composite(canvas, shadow_img)
    draw = ImageDraw.Draw(canvas)

    # Card background (frosted glass dark container)
    draw.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=16 * SCALE,
        fill=(17, 24, 39, 240),
        outline=(255, 255, 255, 28),
        width=1 * SCALE,
    )

    # Card Window Header Bar
    header_h = 42 * SCALE
    draw.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + header_h],
        radius=16 * SCALE,
        fill=(31, 41, 55, 240),
    )
    # Square bottom edges of header
    draw.rectangle(
        [card_x, card_y + header_h - 12 * SCALE, card_x + card_w, card_y + header_h],
        fill=(31, 41, 55, 240),
    )
    draw.line(
        [(card_x, card_y + header_h), (card_x + card_w, card_y + header_h)],
        fill=(255, 255, 255, 20),
        width=1 * SCALE,
    )

    # Mac style window dots
    dot_y = card_y + 16 * SCALE
    draw.ellipse(
        [card_x + 18 * SCALE, dot_y, card_x + 28 * SCALE, dot_y + 10 * SCALE], fill=(239, 68, 68)
    )
    draw.ellipse(
        [card_x + 36 * SCALE, dot_y, card_x + 46 * SCALE, dot_y + 10 * SCALE], fill=(245, 158, 11)
    )
    draw.ellipse(
        [card_x + 54 * SCALE, dot_y, card_x + 64 * SCALE, dot_y + 10 * SCALE], fill=(16, 185, 129)
    )

    # Header window title & status
    draw.text(
        (card_x + 80 * SCALE, card_y + 12 * SCALE),
        "Agentic Assistant",
        fill=(243, 244, 246),
        font=font_badge,
    )
    # Online badge
    draw.ellipse(
        [
            card_x + card_w - 110 * SCALE,
            dot_y + 1 * SCALE,
            card_x + card_w - 102 * SCALE,
            dot_y + 9 * SCALE,
        ],
        fill=(34, 197, 94),
    )
    draw.text(
        (card_x + card_w - 94 * SCALE, card_y + 13 * SCALE),
        "State: Self-Correcting",
        fill=(156, 163, 175),
        font=font_code,
    )

    # --- CHAT CONTENT INSIDE MOCKUP ---
    chat_y = card_y + header_h + 20 * SCALE

    # 1. User Message (Right-aligned bubble)
    user_bubble_w = 400 * SCALE
    user_bubble_h = 56 * SCALE
    user_bubble_x = card_x + card_w - user_bubble_w - 20 * SCALE

    draw.rounded_rectangle(
        [user_bubble_x, chat_y, user_bubble_x + user_bubble_w, chat_y + user_bubble_h],
        radius=12 * SCALE,
        fill=(79, 70, 229, 240),
        outline=(129, 140, 248, 100),
        width=1 * SCALE,
    )
    draw.text(
        (user_bubble_x + 16 * SCALE, chat_y + 10 * SCALE),
        "What is your return policy if an item arrives",
        fill=(255, 255, 255),
        font=font_body,
    )
    draw.text(
        (user_bubble_x + 16 * SCALE, chat_y + 30 * SCALE),
        "damaged or broken in transit?",
        fill=(255, 255, 255),
        font=font_body,
    )

    # 2. Agent Decision Trace Pill (Self-Correction Loop Badge)
    trace_y = chat_y + user_bubble_h + 16 * SCALE
    trace_w = card_w - 40 * SCALE
    trace_h = 32 * SCALE
    trace_x = card_x + 20 * SCALE
    draw.rounded_rectangle(
        [trace_x, trace_y, trace_x + trace_w, trace_y + trace_h],
        radius=8 * SCALE,
        fill=(30, 41, 59, 180),
        outline=(56, 189, 248, 80),
        width=1 * SCALE,
    )
    draw.text(
        (trace_x + 12 * SCALE, trace_y + 8 * SCALE),
        "⚡ Scope: SAFE  •  Retrieved 4 chunks  •  Reranked top 2  •  Confidence: 98%",
        fill=(56, 189, 248),
        font=font_code,
    )

    # 3. Agent Response Bubble
    agent_y = trace_y + trace_h + 16 * SCALE
    agent_w = card_w - 40 * SCALE
    agent_h = 240 * SCALE
    agent_x = card_x + 20 * SCALE

    draw.rounded_rectangle(
        [agent_x, agent_y, agent_x + agent_w, agent_y + agent_h],
        radius=12 * SCALE,
        fill=(31, 41, 55, 220),
        outline=(75, 85, 99, 120),
        width=1 * SCALE,
    )

    # Agent header row inside bubble
    draw.ellipse(
        [agent_x + 16 * SCALE, agent_y + 14 * SCALE, agent_x + 26 * SCALE, agent_y + 24 * SCALE],
        fill=(99, 102, 241),
    )
    draw.text(
        (agent_x + 34 * SCALE, agent_y + 10 * SCALE),
        "Customer Support Agent",
        fill=(249, 250, 251),
        font=font_body_bold,
    )
    draw.text(
        (agent_x + 220 * SCALE, agent_y + 12 * SCALE),
        "Grounded response",
        fill=(52, 211, 153),
        font=font_code,
    )

    # Answer text lines
    ans_lines = [
        "If your item arrives damaged, you are eligible for an immediate",
        "free replacement or full refund within 30 days of delivery.",
        "We automatically generate prepaid return carrier labels once verified.",
    ]
    ans_y = agent_y + 42 * SCALE
    for line in ans_lines:
        draw.text((agent_x + 16 * SCALE, ans_y), line, fill=(229, 231, 235), font=font_body)
        ans_y += 24 * SCALE

    # Citations block
    cite_box_y = ans_y + 10 * SCALE
    draw.text(
        (agent_x + 16 * SCALE, cite_box_y),
        "Verified Evidence & Citations:",
        fill=(156, 163, 175),
        font=font_small_bold,
    )

    # Citation pills
    c1_y = cite_box_y + 22 * SCALE
    # Pill 1
    p1_w = 230 * SCALE
    draw.rounded_rectangle(
        [agent_x + 16 * SCALE, c1_y, agent_x + 16 * SCALE + p1_w, c1_y + 28 * SCALE],
        radius=6 * SCALE,
        fill=(17, 24, 39, 255),
        outline=(99, 102, 241, 140),
        width=1 * SCALE,
    )
    draw.text(
        (agent_x + 26 * SCALE, c1_y + 6 * SCALE),
        "📄 KB §4.1: Damaged Goods Policy",
        fill=(199, 210, 254),
        font=font_code,
    )

    # Pill 2
    p2_x = agent_x + 26 * SCALE + p1_w
    p2_w = 230 * SCALE
    draw.rounded_rectangle(
        [p2_x, c1_y, p2_x + p2_w, c1_y + 28 * SCALE],
        radius=6 * SCALE,
        fill=(17, 24, 39, 255),
        outline=(16, 185, 129, 140),
        width=1 * SCALE,
    )
    draw.text(
        (p2_x + 10 * SCALE, c1_y + 6 * SCALE),
        "📦 Shipping §2: Prepaid Return Labels",
        fill=(167, 243, 208),
        font=font_code,
    )

    # Verification Badges at bottom of agent card
    badge_y = c1_y + 38 * SCALE
    v_badges = [
        ("✓ Grounded Claims", (16, 185, 129)),
        ("✓ Reranker Score: 0.94", (56, 189, 248)),
        ("✓ Zero Hallucination", (168, 85, 247)),
    ]
    bx = agent_x + 16 * SCALE
    for b_text, b_col in v_badges:
        draw.text((bx, badge_y), b_text, fill=b_col, font=font_small_bold)
        bx += 160 * SCALE

    # Downscale from 2560x1280 to 1280x640 with Lanczos
    final_img = canvas.resize((1280, 640), Image.Resampling.LANCZOS)

    # Ensure assets directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path, "PNG", optimize=True)
    print(f"Successfully created {output_path} with size {final_img.size}")


if __name__ == "__main__":
    create_social_preview()
