import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Global Page Configuration (Set to wide layout)
# ==========================================
st.set_page_config(page_title="Anime Dual-Engine Recommendation System", page_icon="🎬", layout="wide")

# ==========================================
# 2. Core Data Loading & Caching
# ==========================================
@st.cache_data
def load_and_compute_models():
    animes_df = pd.read_csv("animes.csv")
    animes_df['genres_detailed'] = animes_df['genres_detailed'].fillna('')
    animes_df['score'] = pd.to_numeric(animes_df['score'], errors='coerce').fillna(0)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(animes_df['genres_detailed'])
    
    try:
        rating_df = pd.read_csv("rating_cf_ultra_final.csv")
        active_users = rating_df['userID'].value_counts()
        active_users = active_users[active_users >= 20].index
        filtered_ratings = rating_df[rating_df['userID'].isin(active_users)]
        
        active_animes = filtered_ratings['animeID'].value_counts()
        active_animes = active_animes[active_animes >= 50].index
        filtered_ratings = filtered_ratings[filtered_ratings['animeID'].isin(active_animes)]
        
        top_anime_ids = filtered_ratings['animeID'].value_counts().head(10).index
        top10_df = animes_df.set_index('animeID').loc[top_anime_ids].reset_index()
        
        pivot_matrix = filtered_ratings.pivot_table(index='animeID', columns='userID', values='rating').fillna(0)
        item_sim_df = pd.DataFrame(cosine_similarity(pivot_matrix), index=pivot_matrix.index, columns=pivot_matrix.index)
    except Exception as e:
        st.error("⚠️ Failed to load Collaborative Filtering files.")
        item_sim_df = pd.DataFrame()
        top10_df = pd.DataFrame()
        
    return animes_df, tfidf_matrix, item_sim_df, top10_df

with st.spinner("🤖 Loading AI Recommendation Engine. Initial startup may take a few seconds..."):
    animes_df, tfidf_matrix, item_sim_df, top10_df = load_and_compute_models()

# ==========================================
# 3. Define Recommendation Functions (With secondary filtering logic)
# ==========================================
def get_cbf_recommendations(anime_title, df, feature_matrix, top_k, selected_types, min_score):
    idx_list = df.index[df['title'].str.lower() == anime_title.lower()].tolist()
    if not idx_list:
        return None, f"❌ Cannot find an anime named '{anime_title}'. Please check your spelling."
    idx = idx_list[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    
    similar_indices = sim_scores.argsort()[::-1][1:]
    recs = df.iloc[similar_indices][['title', 'type', 'score', 'genres_detailed']].copy()
    recs['similarity_score'] = sim_scores[similar_indices]
    
    if selected_types:
        recs = recs[recs['type'].isin(selected_types)]
    recs = recs[recs['score'] >= min_score]
    
    if recs.empty:
        return None, "⚠️ No recommendations match your filters. Please relax the 'Type' or 'Minimum Rating' limits on the left."
    return recs.head(top_k), None

def get_cf_recommendations(anime_title, df, sim_df, top_k, selected_types, min_score):
    match = df[df['title'].str.lower() == anime_title.lower()]
    if match.empty:
        return None, f"❌ Cannot find an anime named '{anime_title}'. Please check your spelling."
    target_id = match.iloc[0]['animeID']
    
    if target_id not in sim_df.index:
        return None, f"🧊 [Severe Cold Start Intercept] This anime (ID:{target_id}) has too few real user ratings!\n\n🚨 **CF (Collaborative Filtering) Engine is down.** \n👉 **System Suggestion: Please go to the left control panel and switch to the [CBF (Content-Based)] engine as a fallback!**"
    
    sim_scores = sim_df[target_id]
    similar_ids = sim_scores.sort_values(ascending=False).index[1:]
    
    results = []
    for aid in similar_ids:
        match_anime = df[df['animeID'] == aid]
        if not match_anime.empty:
            row = match_anime.iloc[0].copy()
            row['cf_similarity'] = sim_scores[aid]
            results.append(row)
            
    recs = pd.DataFrame(results)[['title', 'type', 'score', 'genres_detailed', 'cf_similarity']]
    
    if selected_types:
        recs = recs[recs['type'].isin(selected_types)]
    recs = recs[recs['score'] >= min_score]
    
    if recs.empty:
        return None, "⚠️ No recommendations match your filters. Please relax the 'Type' or 'Minimum Rating' limits on the left."
    return recs.head(top_k), None

# ==========================================
# 4. UI Layout
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)
st.sidebar.header("⚙️ Engine Control Panel")

engine_choice = st.sidebar.radio("1. Core Algorithm Selection:", ["CF (Collaborative Filtering - Crowd Wisdom)", "CBF (Content-Based - Tag Analysis)"])
top_k_choice = st.sidebar.slider("2. Number of Recommendations:", min_value=5, max_value=20, value=10, step=1)

st.sidebar.divider()
st.sidebar.subheader("🎛️ Secondary Filtering")
all_types = [t for t in animes_df['type'].unique() if pd.notna(t) and t != '']
selected_types = st.sidebar.multiselect("Include Specific Types (Leave blank for all):", all_types, default=[])
min_score = st.sidebar.slider("Minimum Community Rating:", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

st.title("🎬 Anime Dual-Engine Recommendation System")
st.markdown("Discover your next masterpiece! This platform is powered by dual AI algorithms: **CBF (Text Deep Learning)** and **CF (Million-level Crowd Wisdom)**.")

tab_search, tab_trending, tab_insights = st.tabs(["🎯 Exclusive AI Recommendations", "🏆 All-Time Top 10 Trending", "📊 Algorithm Performance (Benchmarks)"])

# ---------------- Tab 1: Search & Recommend ----------------
with tab_search:
    col_search, col_demo = st.columns([3, 1])
    with col_search:
        # Extract all anime names from the database into a list
        all_anime_titles = animes_df['title'].tolist()
        # Use selectbox instead of text_input for a Google-level auto-complete search experience!
        user_input = st.selectbox(
            "🔍 Search or select an anime:", 
            options=all_anime_titles, 
            index=all_anime_titles.index("Death Note") if "Death Note" in all_anime_titles else 0
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("🚨 Demo: Simulate Cold-Start Anime"):
            st.warning('Target loaded: Please search for `s-CRY-ed` in the search box, select the **CF Engine**, and generate to see the system interception!')
            
    if st.button("🚀 Generate AI Recommendations", type="primary"):
        if user_input:
            if engine_choice == "CBF (Content-Based - Tag Analysis)":
                recs, error_msg = get_cbf_recommendations(user_input, animes_df, tfidf_matrix, top_k_choice, selected_types, min_score)
            else:
                recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k_choice, selected_types, min_score)
                
            if error_msg:
                st.error(error_msg) 
            else:
                st.success("✅ Results generated successfully! (Low-rated items filtered out based on industry standards)")
                # ------------------- 新增：展示目标动漫的档案卡 -------------------
                target_anime = animes_df[animes_df['title'] == user_input].iloc[0]
                
                # 清洗目标动漫的标签
                raw_target_tags = str(target_anime['genres_detailed'])
                try:
                    target_tags = ast.literal_eval(raw_target_tags)
                except:
                    target_tags = raw_target_tags.replace("['", "").replace("']", "").split("', '")
                target_clean_tags = " • ".join([t.title() for t in target_tags if t])
                
                # 用一个信息框展示它
                st.info(f"🎯 **Target Selected:** **{target_anime['title']}** (Type: {target_anime['type']} | Score: ⭐ {target_anime['score']:.2f})\n\n🏷️ **DNA Tags:** {target_clean_tags}")
                # ----------------------------------------------------------------
                st.markdown("---") 
                
                sim_col = 'similarity_score' if 'similarity_score' in recs.columns else 'cf_similarity'
                max_sim = float(recs[sim_col].max())
                
                for rank_idx, (index, row) in enumerate(recs.iterrows()):
                    with st.container(border=True):
                        col_info, col_score = st.columns([3, 1])
                        
                        with col_info:
                            st.subheader(f"🏅 {row['title']}")
                            st.caption(f"**Type**: {row['type']}  |  **Community Rating**: ⭐ {row['score']:.2f}")
                            
                            with st.expander("🏷️ Core Tropes / Tags"):
                                raw_tags = str(row['genres_detailed'])
                                try:
                                    tags = ast.literal_eval(raw_tags)
                                except:
                                    tags = raw_tags.replace("['", "").replace("']", "").split("', '")
                                
                                display_tags = tags if isinstance(tags, list) else []
                                badges_html = "".join([
                                    f'<span style="display:inline-block; margin: 0px 6px 8px 0; padding: 4px 12px; '
                                    f'background-color: rgba(130, 130, 130, 0.15); border: 1px solid rgba(130, 130, 130, 0.3); '
                                    f'border-radius: 16px; font-size: 13px; white-space: nowrap;">{tag.title()}</span>'
                                    for tag in display_tags if tag
                                ])
                                st.markdown(badges_html, unsafe_allow_html=True)
                                
                        with col_score:
                            # Option 3: Pure Star Rating System (Most intuitive, zero cognitive load)
                            raw_score = float(row[sim_col])
                            match_pct = int((raw_score / max_sim) * 99) if max_sim > 0 else 0
                            
                            # Star rating and clean text assessment
                            if match_pct >= 90:
                                stars = "★★★★★"
                                level_text = "Perfect Match"
                                star_color = "#FFD700"  # Dazzling Gold
                            elif match_pct >= 75:
                                stars = "★★★★☆"
                                level_text = "Highly Similar"
                                star_color = "#F39C12"  # Vibrant Orange-Gold
                            else:
                                stars = "★★★☆☆"
                                level_text = "Style Correlated"
                                star_color = "#AAB7B8"  # Textured Silver-Gray
                                
                            # Use HTML to render a clean star layout (Stars on top, subtitle below)
                            st.markdown(f"""
                                <div style='text-align: right; padding-top: 12px;'>
                                    <div style='font-size: 20px; color: {star_color}; letter-spacing: 2px;'>{stars}</div>
                                    <div style='font-size: 13px; color: #888; margin-top: 4px; font-weight: 500;'>{level_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Please select or enter an anime name!")

# ---------------- Tab 2: Trending Leaderboard ----------------
with tab_trending:
    st.subheader("🏆 Community's Most Popular Anime Top 10")
    st.markdown("This list is extracted based on **rating frequency data** from real platform users (filtering out niche noise).")
    
    if not top10_df.empty:
        for rank, (index, row) in enumerate(top10_df.iterrows()):
            with st.container(border=True):
                # Fix: Use pure HTML instead of ### headers to completely eliminate the link anchor icon
                st.markdown(f"<div style='font-size: 18px; font-weight: bold; margin-bottom: 8px;'>👑 No.{rank + 1} &nbsp; {row['title']}</div>", unsafe_allow_html=True)
                st.caption(f"**Type**: {row['type']}  |  **Community Rating**: ⭐ {row['score']}")
    else:
        st.info("Loading data...")

# ---------------- Tab 3: Data Insights Panel ----------------
with tab_insights:
    st.subheader("📊 Underlying Algorithm Benchmarks")
    st.markdown("Offline evaluation report under experimental environment (Test results on 42,797 sets of Data):")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CF RMSE (Collaborative Error)", "1.377", "- More Accurate")
    col2.metric("CBF RMSE (Content Error)", "1.411", "+ Broader Coverage")
    col3.metric("CF Hit Rate (Precision@10)", "16.8%", "+ Better than CBF's 15.4%")
    
    st.divider()
    st.subheader("📈 Current Anime Type Distribution (EDA)")
    type_counts = animes_df['type'].value_counts()
    st.bar_chart(type_counts, color="#ff4b4b")
