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

st.markdown("---")
st.caption("Food Verse AI · Estimates are informational only. Never rely on image analysis alone for allergen safety, food spoilage, commercial compliance, or medical nutrition decisions.")
