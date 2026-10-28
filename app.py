from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

APP_NAME = "Food Verse AI"
st.set_page_config(page_title=APP_NAME, page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- Theme & utilities -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
:root { --fv-ink:#18212f; --fv-muted:#64748b; --fv-green:#1d9b72; --fv-lime:#d6f36a; --fv-purple:#7c5cff; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 90% 0%, rgba(214,243,106,.20), transparent 28%), linear-gradient(180deg,#f7faf7 0%,#f5f7fb 100%); color:var(--fv-ink); }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; letter-spacing:-.03em; }
.hero { padding:1.5rem 1.7rem; border-radius:24px; background:linear-gradient(120deg,#152c2a,#1d9b72 72%,#a7d957); color:white; box-shadow:0 14px 35px rgba(23,83,64,.16); }
.hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; opacity:.82; }
.hero h1 { color:white; font-size:clamp(2rem,4vw,3.2rem); margin:.3rem 0; }
.hero p { max-width:780px; opacity:.9; margin:0; font-size:1.02rem; }
.pill { display:inline-block; background:#e5f8ef; color:#126b50; border:1px solid #c5efdc; border-radius:999px; padding:.28rem .7rem; font-size:.78rem; font-weight:700; margin:.2rem .25rem .2rem 0; }
.metric-card { background:white; border:1px solid #e8eee9; padding:1rem 1.1rem; border-radius:18px; box-shadow:0 5px 18px rgba(22,40,32,.04); min-height:100px; }
.metric-label { color:#64748b; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.metric-value { font-family:'Outfit',sans-serif; font-size:1.8rem; font-weight:800; color:#183a30; }
.muted { color:#64748b; }
div[data-testid="stMetric"] { background:white; padding:1rem; border-radius:16px; border:1px solid #e8eee9; }
.stButton>button { border-radius:12px; font-weight:700; }
section[data-testid="stSidebar"] { background:#eff5ef; }
.small-note { font-size:.82rem; color:#64748b; }
</style>
""", unsafe_allow_html=True)

def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="small-note">{note}</div></div>', unsafe_allow_html=True)

def safe_image(upload, max_pixels: int = 20_000_000) -> Image.Image:
    img = Image.open(upload).convert("RGB")
    if img.width * img.height > max_pixels:
        img.thumbnail((2500, 2500))
    return img

def split_items(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;\n]", text) if x.strip()]

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def get_ai_provider() -> tuple[str, str]:
    provider = st.session_state.get("provider", "Demo mode")
    key = st.session_state.get("api_key", "").strip()
    if provider == "Gemini" and key:
        return provider, key
    if provider == "Groq" and key:
        return provider, key
    return "Demo mode", ""

def call_vision(image: Image.Image, task: str, context: str = "") -> dict[str, Any]:
    """Use a configured vision-capable provider; return demo response when no key is set."""
    provider, key = get_ai_provider()
    if provider == "Gemini":
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            prompt = task + "\nReturn only valid JSON. " + context
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            return json.loads(response.text)
        except Exception as exc:
            st.warning(f"Gemini analysis failed; using demonstration output. Details: {type(exc).__name__}")
    elif provider == "Groq":
        # Groq's hosted vision models generally accept image URLs/base64 through OpenAI-compatible APIs.
        # To keep the app portable and avoid oversized payloads, convert the image to a data URL.
        try:
            import base64
            from io import BytesIO
            from groq import Groq
            from openai import OpenAI
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=82)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role":"user","content":[
                    {"type":"text","text":task + "\nReturn only valid JSON. " + context},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]}],
                temperature=0.2,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content or "{}"
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.I|re.M).strip()
            return json.loads(raw)
        except Exception as exc:
            st.warning(f"Groq analysis failed; using demonstration output. Details: {type(exc).__name__}")
    return demo_vision(task, context)

def demo_vision(task: str, context: str = "") -> dict[str, Any]:
    t = task.lower()
    if "freshness" in t or "spoilage" in t:
        return {
            "item": "Produce item (demo estimate)",
            "freshness_stage": "Moderately Fresh",
            "visual_observations": ["No reliable species identification in demo mode", "Image-only inspection cannot confirm microbial safety"],
            "risk_flags": ["Check smell, texture, storage history, and expiry date"],
            "advice": "If it is perishable and has been in the temperature danger zone for too long, discard it. Do not taste food to test safety.",
            "storage_tips": ["Keep refrigerated perishables at 4°C / 40°F or below", "Separate raw meat from ready-to-eat foods"]
        }
    if "kitchen" in t or "hygiene" in t:
        return {
            "observations": [
                {"indicator":"Hair restraint visible","status":"Needs verification","severity":"Medium","evidence":"Demo mode cannot inspect the uploaded scene"},
                {"indicator":"Glove use / hand hygiene","status":"Needs verification","severity":"Medium","evidence":"Verify handwashing and task-appropriate glove changes"},
                {"indicator":"Sanitization","status":"Needs verification","severity":"High","evidence":"Check food-contact surfaces and sanitizer concentration"},
                {"indicator":"Raw-to-ready cross-contamination","status":"Needs verification","severity":"High","evidence":"Keep raw poultry, utensils and juices separate from produce"}
            ],
            "score": 0,
            "disclaimer": "No actual visual detection was performed in demo mode."
        }
    return {
        "dish_name": "Example vegetable grain bowl (demo)",
        "cuisine": "Fusion / illustrative",
        "ingredients": ["Cooked grains", "Mixed vegetables", "Legumes", "Dressing"],
        "portion_g": 350,
        "nutrition": {"Calories_kcal": 420, "Protein_g": 18, "Carbohydrates_g": 58, "Fats_g": 12, "Fiber_g": 11},
        "allergens": ["Possible sesame / soy depending on dressing"],
        "health_score": 78,
        "notes": "Illustrative demo estimate, not derived from the uploaded image."
    }

def render_nutrition(data: dict[str, Any]) -> None:
    st.subheader(str(data.get("dish_name", "Dish analysis")))
    st.caption(f"Cuisine: {data.get('cuisine', 'Uncertain')} · Estimated portion: {data.get('portion_g', '—')} g")
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, str):
        ingredients = split_items(ingredients)
    st.write("**Likely ingredients:** " + (", ".join(map(str, ingredients)) if ingredients else "Not confidently identified"))
    n = data.get("nutrition", {}) or {}
    fields = [
        ("Calories", "Calories_kcal", "kcal"),
        ("Protein", "Protein_g", "g"),
        ("Carbohydrates", "Carbohydrates_g", "g"),
        ("Fats", "Fats_g", "g"),
        ("Fiber", "Fiber_g", "g"),
    ]
    cols = st.columns(5)
    for col, (label, key, unit) in zip(cols, fields):
        value = n.get(key, 0)
        try: value = float(value)
        except (TypeError, ValueError): value = 0
        with col: metric_card(label, f"{value:g} {unit}")
    macro_data = pd.DataFrame([
        {"Macronutrient":"Protein", "Grams":float(n.get("Protein_g",0) or 0)},
        {"Macronutrient":"Carbohydrates", "Grams":float(n.get("Carbohydrates_g",0) or 0)},
        {"Macronutrient":"Fats", "Grams":float(n.get("Fats_g",0) or 0)},
        {"Macronutrient":"Fiber", "Grams":float(n.get("Fiber_g",0) or 0)},
    ])
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(px.bar(macro_data, x="Macronutrient", y="Grams", color="Macronutrient", title="Macro breakdown (grams)"), use_container_width=True)
    with c2:
        score = clamp(float(data.get("health_score", 0) or 0), 1, 100)
        st.metric("Composite health score", f"{score:.0f}/100")
        if score >= 75: st.success("Higher score in this illustrative scoring rubric")
        elif score >= 50: st.warning("Moderate score — review balance and portion size")
        else: st.error("Lower score — consider a more balanced meal")
        allergens = data.get("allergens", [])
        if isinstance(allergens, str): allergens = split_items(allergens)
        if allergens: st.warning("Allergen watch: " + ", ".join(map(str, allergens)))
        st.caption("Nutrition and health scores are estimates. Confirm ingredients, portion weights, and allergen labels before making decisions.")

# -------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## 🥗 Food Verse AI")
    st.caption("Multimodal Food Intelligence Agent")
    st.markdown("---")
    st.markdown("### AI configuration")
    provider = st.selectbox("Provider", ["Demo mode", "Gemini", "Groq"], index=0, key="provider")
    api_key = st.text_input("API key", type="password", key="api_key", placeholder="Paste key for live analysis")
    st.caption("Keys stay in this session and are not written to a database. For deployed apps, prefer Streamlit secrets.")
    if provider != "Demo mode" and not api_key:
        st.info("No key entered — the app will use demo output.")
    st.markdown("---")
    page = st.radio("Navigate pillars", [
        "Food Recognition & Nutrition",
        "Freshness & Spoilage",
        "Kitchen Hygiene & Safety",
        "Smart Recipe Assistant",
        "Personalized Meal Planner",
    ], label_visibility="visible")
    st.markdown("---")
    st.caption("Skill Wallet · Capstone Project 3")

# -------------------------------- Main header --------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Skill Wallet · Capstone Project 3</div>
  <h1>Food Verse AI</h1>
  <p>One workspace for food recognition, nutrition insights, freshness checks, kitchen safety, creative cooking and personalized meal planning.</p>
  <div style="margin-top:.8rem"><span class="pill">Vision AI</span><span class="pill">Nutrition</span><span class="pill">Food Safety</span><span class="pill">Meal Planning</span></div>
</div>
""", unsafe_allow_html=True)
st.write("")

if page == "Food Recognition & Nutrition":
    st.header("🍽️ Food Recognition & Nutrition Audit")
    st.write("Upload a meal photo to identify likely ingredients and estimate its nutrition profile.")
    upload = st.file_uploader("Upload meal / dish photo", type=["png","jpg","jpeg","webp"], key="food_img")
    if upload:
        try:
            img = safe_image(upload)
            left, right = st.columns([.85, 1.15])
            with left: st.image(img, caption="Uploaded meal", use_container_width=True)
            with right:
                st.markdown("#### Analysis controls")
                portion = st.number_input("Approximate portion weight (g)", min_value=20, max_value=2500, value=350, step=25)
                extra = st.text_input("Known ingredients / context (optional)", placeholder="e.g., homemade, no added sugar")
                if st.button("Analyze food & nutrition", type="primary", use_container_width=True):
                    prompt = """You are a cautious food-science assistant. Inspect the image and return JSON with keys: dish_name, cuisine, ingredients (array), portion_g (number), nutrition (object with Calories_kcal, Protein_g, Carbohydrates_g, Fats_g, Fiber_g), allergens (array), health_score (integer 1-100), notes. Estimate visible dish only; do not invent certainty. Flag likely allergens and mention uncertainty. Nutrition values should correspond to the given portion."""
                    with st.spinner("Analyzing dish…"):
                        result = call_vision(img, prompt, f"User portion estimate: {portion} g. Extra context: {extra}")
                    if result.get("nutrition") is None: result["nutrition"] = {}
                    result["portion_g"] = result.get("portion_g") or portion
                    st.session_state["nutrition_result"] = result
            if "nutrition_result" in st.session_state:
                st.divider()
                render_nutrition(st.session_state["nutrition_result"])
                st.info(st.session_state["nutrition_result"].get("notes", ""))
        except Exception as e:
            st.error(f"Could not read or analyze this image: {e}")
    else:
        st.info("Try a clear, well-lit photo with the whole serving visible.")

elif page == "Freshness & Spoilage":
    st.header("🧪 Food Freshness & Spoilage Diagnostic")
    st.warning("Image-based freshness checks are not a food-safety test. Pathogens and toxins may be present without visible changes.")
    upload = st.file_uploader("Upload fruit, vegetable, meat or dairy photo", type=["png","jpg","jpeg","webp"], key="fresh_img")
    item_type = st.selectbox("Food category", ["Fruit", "Vegetable", "Raw meat / poultry", "Fish / seafood", "Dairy", "Cooked leftovers", "Other"])
    storage = st.selectbox("Known storage history", ["Unknown", "Refrigerated continuously", "Room temperature briefly", "Left out for 2+ hours"])
    if upload:
        try:
            img = safe_image(upload)
            st.image(img, caption="Uploaded item", width=440)
            if st.button("Inspect freshness", type="primary"):
                prompt = """Assess visible food freshness conservatively. Return JSON keys: item, freshness_stage (Fresh | Moderately Fresh | Spoiled), visual_observations (array), risk_flags (array), advice, storage_tips (array). Do not claim pathogens can be detected visually. If uncertain, say so and recommend safe handling."""
                with st.spinner("Inspecting visible signs…"):
                    result = call_vision(img, prompt, f"Category: {item_type}; storage history: {storage}")
                st.session_state["fresh_result"] = result
            if "fresh_result" in st.session_state:
                result = st.session_state["fresh_result"]
                stage = result.get("freshness_stage", "Uncertain")
                if stage == "Fresh": st.success(f"Visible assessment: {stage}")
                elif stage == "Moderately Fresh": st.warning(f"Visible assessment: {stage}")
                else: st.error(f"Visible assessment: {stage}")
                st.write("**Item:**", result.get("item", item_type))
                st.write("**Observations:**")
                for x in result.get("visual_observations", []): st.write(f"- {x}")
                st.write("**Risk flags:**")
                for x in result.get("risk_flags", []): st.write(f"- {x}")
                st.write("**Advice:**", result.get("advice", "When in doubt, discard high-risk perishables."))
                st.write("**Storage tips:**")
                for x in result.get("storage_tips", []): st.write(f"- {x}")
        except Exception as e: st.error(f"Image processing failed: {e}")
    else:
        st.caption("Look for visible discoloration, wrinkling, slime or mold, but do not use appearance alone to decide safety.")

elif page == "Kitchen Hygiene & Safety":
    st.header("🧼 Kitchen Hygiene & Food Safety Compliance")
    st.write("Upload a kitchen image or a representative video frame. This is a screening aid, not a formal inspection.")
    media_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
    if media_type == "Image":
        upload = st.file_uploader("Upload kitchen image", type=["png","jpg","jpeg","webp"], key="kitchen_img")
        if upload:
            try:
                img = safe_image(upload)
                st.image(img, caption="Kitchen scene", use_container_width=True)
                if st.button("Audit kitchen hygiene", type="primary"):
                    prompt = """Review a commercial kitchen image for visible hygiene indicators. Return JSON keys: observations (array of objects with indicator, status, severity, evidence), score (integer 0-100), disclaimer. Evaluate hair restraint, hand/glove hygiene, surface sanitization and raw/ready-to-eat cross contamination. If not visible, use 'Not observable' rather than assuming compliance."""
                    with st.spinner("Reviewing visible indicators…"):
                        st.session_state["kitchen_result"] = call_vision(img, prompt)
            except Exception as e: st.error(f"Image processing failed: {e}")
    else:
        video = st.file_uploader("Upload short kitchen video", type=["mp4","mov","avi","webm"], key="kitchen_video")
        if video:
            st.video(video)
            st.info("For this lightweight deployment, video is previewed locally in the session. Upload a representative frame as an image for AI analysis.")
    if "kitchen_result" in st.session_state:
        result = st.session_state["kitchen_result"]
        score = clamp(float(result.get("score", 0) or 0), 0, 100)
        st.metric("Kitchen hygiene compliance score", f"{score:.0f}%")
        st.progress(score / 100)
        obs = result.get("observations", [])
        if obs:
            df = pd.DataFrame(obs)
            st.dataframe(df, use_container_width=True, hide_index=True)
            violations = [o for o in obs if str(o.get("status","")).lower() not in ["compliant","pass","ok"]]
            st.write(f"**Items requiring review:** {len(violations)}")
        st.caption(result.get("disclaimer", "Visual review cannot verify actual sanitation, temperatures or staff procedures."))

elif page == "Smart Recipe Assistant":
    st.header("👩‍🍳 Smart Kitchen Recipe Assistant")
    ingredients_text = st.text_area("What's in your pantry?", placeholder="Chickpeas, spinach, rice, tomato, yogurt…", height=110)
    c1, c2, c3 = st.columns(3)
    with c1: diet = st.selectbox("Diet preference", ["No preference", "Vegetarian", "Vegan", "High-protein", "Keto"])
    with c2: time_limit = st.selectbox("Time available", ["15 minutes", "30 minutes", "45 minutes", "60+ minutes"])
    with c3: servings = st.number_input("Servings", min_value=1, max_value=12, value=2)
    avoid = st.text_input("Avoid / allergy constraints", placeholder="e.g., peanuts, dairy")
    if st.button("Create recipes", type="primary", use_container_width=True):
        items = split_items(ingredients_text)
        if not items:
            st.error("Add at least one pantry ingredient first.")
        else:
            provider, key = get_ai_provider()
            if provider == "Gemini":
                try:
                    from google import genai
                    client = genai.Client(api_key=key)
                    prompt = f"""Create 3 practical recipes using these available ingredients: {items}. Diet: {diet}. Time: {time_limit}. Servings: {servings}. Avoid: {avoid}. Return JSON with recipes array. Each recipe needs title, summary, cooking_time, difficulty, ingredients array, steps array, substitutions array, nutrition_note."""
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config={"response_mime_type":"application/json","temperature":0.5})
                    data = json.loads(response.text)
                except Exception as exc:
                    st.warning(f"Recipe generation failed; showing demo recipes ({type(exc).__name__}).")
                    data = demo_recipes(items, diet, servings)
            elif provider == "Groq":
                try:
                    from groq import Groq
                    client = Groq(api_key=key)
                    prompt = f"""Create 3 practical recipes using ingredients {items}; diet {diet}; time {time_limit}; servings {servings}; avoid {avoid}. Return valid JSON object with recipes array; each recipe has title, summary, cooking_time, difficulty, ingredients, steps, substitutions, nutrition_note."""
                    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], response_format={"type":"json_object"}, temperature=0.5)
                    data = json.loads(response.choices[0].message.content)
                except Exception as exc:
                    st.warning(f"Recipe generation failed; showing demo recipes ({type(exc).__name__}).")
                    data = demo_recipes(items, diet, servings)
            else:
                data = demo_recipes(items, diet, servings)
            st.session_state["recipes"] = data.get("recipes", [])
    for recipe in st.session_state.get("recipes", []):
        with st.expander(f"{recipe.get('title','Recipe')} · {recipe.get('cooking_time','')} · {recipe.get('difficulty','')}"):
            st.write(recipe.get("summary", ""))
            st.write("**Ingredients**")
            for x in recipe.get("ingredients", []): st.write(f"- {x}")
            st.write("**Method**")
            for i, x in enumerate(recipe.get("steps", []), 1): st.write(f"{i}. {x}")
            st.write("**Substitutions:** " + "; ".join(recipe.get("substitutions", [])))
            st.caption(recipe.get("nutrition_note", "Nutrition depends on exact ingredients and portions."))

elif page == "Personalized Meal Planner":
    st.header("📅 Personalized Meal Planner")
    st.write("Estimate daily energy needs and build an editable 7-day plan. Targets are approximate, not medical advice.")
    with st.form("profile_form"):
        a, b, c = st.columns(3)
        with a: age = st.number_input("Age (years)", 18, 100, 25)
        with b: gender = st.selectbox("Sex used for equation", ["Female", "Male"])
        with c: height = st.number_input("Height (cm)", 120, 230, 170)
        d, e, f = st.columns(3)
        with d: weight = st.number_input("Weight (kg)", 30.0, 300.0, 70.0, step=0.5)
        with e: activity = st.selectbox("Activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
        with f: goal = st.selectbox("Fitness goal", ["Maintain", "Fat Loss", "Muscle Gain"])
        diet = st.selectbox("Dietary pattern", ["Balanced", "Vegan", "Vegetarian", "Keto", "Diabetic-Friendly"])
        allergies = st.text_input("Allergies / exclusions", placeholder="e.g., peanuts, shellfish")
        submit = st.form_submit_button("Calculate targets & build 7-day plan", type="primary", use_container_width=True)
    if submit:
        # Mifflin-St Jeor equation; sex-specific coefficient is an approximation.
        bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)
        factors = {"Sedentary":1.2, "Lightly active":1.375, "Moderately active":1.55, "Very active":1.725, "Extra active":1.9}
        tdee = bmr * factors[activity]
        target = tdee - 400 if goal == "Fat Loss" else (tdee + 250 if goal == "Muscle Gain" else tdee)
        target = int(clamp(target, 1200, 4500))
        st.session_state["meal_profile"] = {"bmr":round(bmr), "tdee":round(tdee), "target":target, "diet":diet, "allergies":allergies, "goal":goal}
        st.session_state["meal_plan"] = build_meal_plan(diet, allergies)
    if "meal_profile" in st.session_state:
        p = st.session_state["meal_profile"]
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("Estimated BMR", f"{p['bmr']} kcal/day", "Resting energy estimate")
        with m2: metric_card("Estimated TDEE", f"{p['tdee']} kcal/day", "Daily energy expenditure")
        with m3: metric_card("Daily target", f"{p['target']} kcal/day", p["goal"])
        st.caption("The Mifflin–St Jeor estimate is a starting point; actual needs vary. Very low targets may be inappropriate for some people. Consult a qualified professional for medical conditions, pregnancy, or therapeutic diets.")
        plan = st.session_state["meal_plan"]
        st.subheader("Your 7-day meal schedule")
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
        grocery = sorted({item for day in plan for item in day["Grocery list"].split(", ") if item})
        st.subheader("🛒 Consolidated grocery list")
        for item in grocery: st.checkbox(item, key=f"grocery_{item}")
        st.download_button("Download meal plan (CSV)", pd.DataFrame(plan).to_csv(index=False).encode("utf-8"), "food_verse_7_day_meal_plan.csv", "text/csv")

def demo_recipes(items: list[str], diet: str, servings: int) -> dict[str, Any]:
    base = items[:5]
    if not base: base = ["mixed vegetables", "grain", "beans"]
    return {"recipes":[
        {"title":"Pantry Power Bowl","summary":"A flexible bowl that uses your available ingredients.","cooking_time":"20 minutes","difficulty":"Easy","ingredients":[f"{x} (as needed)" for x in base]+[f"Oil, salt and spices to taste"],"steps":["Wash and prep the ingredients.","Cook grains or legumes if they are not already prepared.","Sauté or roast vegetables until tender.","Combine, season, and serve in portions."],"substitutions":["Use lemon juice instead of vinegar","Use beans or lentils for extra protein"],"nutrition_note":f"Demo recipe for {servings} serving(s); adapt to {diet} and verify ingredient labels."},
        {"title":"Quick Skillet Mix","summary":"A one-pan meal with flexible seasoning.","cooking_time":"15–25 minutes","difficulty":"Easy","ingredients":[f"{x} (as needed)" for x in base[:4]],"steps":["Heat a pan and add a little oil or water.","Add ingredients that need the longest cooking first.","Cook until all ingredients reach safe doneness.","Season and serve."],"substitutions":["Swap in any similar seasonal vegetable","Use herbs and spices to change the flavor"],"nutrition_note":"Demo estimate only; not a laboratory nutrition calculation."},
        {"title":"Warm Comfort Soup","summary":"Turn pantry staples into a warming soup.","cooking_time":"25–30 minutes","difficulty":"Easy","ingredients":[f"{x} (as needed)" for x in base[:4]]+["Water or low-sodium stock"],"steps":["Chop ingredients into similar-sized pieces.","Simmer with stock until thoroughly cooked and tender.","Blend partially if desired.","Taste, season, and serve hot."],"substitutions":["Use water plus herbs if stock is unavailable","Use lentils or beans to make it heartier"],"nutrition_note":"Demo recipe; nutrition depends on actual quantities and additions."}
    ]}

def build_meal_plan(diet: str, allergies: str) -> list[dict[str, str]]:
    menus = {
        "Balanced": [("Oats, yogurt and fruit","Dal, rice and vegetables","Grilled paneer/tofu, roti and salad","Fruit and nuts"),
                     ("Egg/tofu scramble and toast","Chickpea salad bowl","Vegetable khichdi with yogurt","Roasted chana"),
                     ("Poha with peas","Rajma rice and salad","Vegetable stir-fry with tofu","Curd or soy yogurt"),
                     ("Besan chilla and fruit","Vegetable pulao with raita","Dal, roti and seasonal vegetables","Buttermilk or fortified soy drink"),
                     ("Overnight oats","Paneer/tofu wrap","Chole with roti and salad","Fruit"),
                     ("Idli, sambar and fruit","Lentil soup and toast","Stuffed vegetables with grains","Yogurt or soy yogurt"),
                     ("Upma with vegetables","Bean bowl with rice","Vegetable curry, roti and dal","Seeds and fruit")],
        "Vegan": [("Oats with soy milk and banana","Chickpea grain bowl","Tofu curry with rice","Fruit and roasted chana"),
                  ("Peanut-free seed butter toast","Lentil soup and bread","Bean chili with rice","Soy yogurt"),
                  ("Poha with peas","Rajma rice and salad","Tofu vegetable stir-fry","Fruit"),
                  ("Besan chilla","Chickpea wrap","Dal, roti and vegetables","Roasted edamame"),
                  ("Overnight oats with soy milk","Quinoa/bean bowl","Chole with roti","Fruit and seeds"),
                  ("Idli and sambar","Lentil salad","Tofu curry with grains","Soy yogurt"),
                  ("Vegetable upma","Bean burrito bowl","Mixed dal and roti","Fruit")],
        "Vegetarian": [("Oats, milk/yogurt and fruit","Dal, rice and vegetables","Paneer curry with roti","Fruit and nuts"),
                       ("Egg/tofu scramble and toast","Chickpea salad bowl","Vegetable khichdi with yogurt","Roasted chana"),
                       ("Poha with peas","Rajma rice and salad","Paneer/tofu stir-fry","Curd"),
                       ("Besan chilla","Vegetable pulao with raita","Dal, roti and vegetables","Buttermilk"),
                       ("Overnight oats","Paneer wrap","Chole with roti","Fruit"),
                       ("Idli and sambar","Lentil soup and toast","Stuffed vegetables with grains","Yogurt"),
                       ("Upma with vegetables","Bean bowl with rice","Vegetable curry, roti and dal","Seeds and fruit")],
        "Keto": [("Paneer/tofu scramble with spinach","Cauliflower rice bowl with tofu","Paneer with sautéed low-carb vegetables","Seeds"),
                 ("Unsweetened yogurt or soy alternative","Salad with tofu and olive oil dressing","Zucchini-paneer skillet","Cucumber"),
                 ("Chia pudding, unsweetened","Tofu lettuce wraps","Cauliflower stir-fry with paneer","A few nuts"),
                 ("Spinach omelet or tofu scramble","Avocado-free salad with paneer","Mushroom-tofu skillet","Cheese or tofu cubes"),
                 ("Yogurt with seeds","Paneer salad bowl","Low-carb vegetable curry","Cucumber"),
                 ("Tofu scramble","Cauliflower bowl","Paneer and greens","Seeds"),
                 ("Chia bowl","Lettuce wraps","Grilled paneer/tofu with greens","Yogurt")],
        "Diabetic-Friendly": [("Oats with plain yogurt and berries","Dal, modest rice portion and vegetables","Tofu/paneer, roti and salad","Nuts or roasted chana"),
                             ("Besan chilla and plain yogurt","Chickpea salad with extra vegetables","Vegetable dalia and curd","Apple with nuts"),
                             ("Vegetable poha with extra peas","Rajma with measured rice and salad","Tofu stir-fry and whole-grain roti","Plain yogurt"),
                             ("Egg/tofu scramble and whole-grain toast","Lentil soup and salad","Dal, roti and vegetables","Seeds and fruit"),
                             ("Plain yogurt, oats and seeds","Bean bowl with measured grains","Paneer/tofu with vegetables","Roasted chana"),
                             ("Idli with extra sambar","Chickpea salad","Mixed dal, roti and vegetables","Fruit with nuts"),
                             ("Vegetable upma, modest portion","Dal and salad bowl","Tofu/paneer, vegetables and roti","Plain yogurt")]
    }
    selected = menus.get(diet, menus["Balanced"])
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    rows = []
    for i, (breakfast,lunch,dinner,snack) in enumerate(selected):
        rows.append({"Day":days[i],"Breakfast":breakfast,"Lunch":lunch,"Dinner":dinner,"Snack":snack,
                     "Grocery list":", ".join(sorted(set(split_items(", ".join([breakfast,lunch,dinner,snack]))))[:8])})
    # Allergy warning: the template is not an allergen-safe meal prescription.
    if allergies.strip():
        for row in rows:
            row["Allergy review"] = f"Review labels and remove: {allergies}. Menu is a template, not verified allergen-free."
    return rows

st.markdown("---")
st.caption("Food Verse AI · Estimates are informational only. Never rely on image analysis alone for allergen safety, food spoilage, commercial compliance, or medical nutrition decisions.")
