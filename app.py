import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Global Page Configuration
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
# 3. Define Recommendation Functions
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
        return None, f"🧊 [Cold Start Intercept] This anime doesn't have enough community ratings yet!\n\n🚨 **CF Engine cannot process this.** \n👉 **Suggestion: Switch to the [CBF (Story DNA)] engine on the left panel to analyze it by plot instead!**"
    
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

# Viewer-Friendly Engine Names
engine_choice = st.sidebar.radio(
    "1. Core Algorithm Selection:", 
    [
        "CF (Community Favorites - Based on User Tastes)", 
        "CBF (Story DNA - Based on Plot & Genres)"
    ],
    help="CF finds what people with similar tastes love. CBF finds anime with the exact same plot tags."
)
top_k_choice = st.sidebar.slider("2. Number of Recommendations:", min_value=5, max_value=20, value=10, step=1)

st.sidebar.divider()
st.sidebar.subheader("🎛️ Secondary Filtering")
all_types = [t for t in animes_df['type'].unique() if pd.notna(t) and t != '']
selected_types = st.sidebar.multiselect("Include Specific Types (Leave blank for all):", all_types, default=[])
min_score = st.sidebar.slider("Minimum Community Rating:", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

st.title("🎬 Anime Dual-Engine Recommendation System")
st.markdown("Discover your next masterpiece! Powered by dual AI algorithms analyzing both **Story DNA** and **Community Wisdom**.")

tab_search, tab_trending, tab_insights = st.tabs(["🎯 Exclusive AI Recommendations", "🏆 All-Time Top 10 Trending", "📊 Algorithm Benchmarks"])

# ---------------- Tab 1: Search & Recommend ----------------
with tab_search:
    all_anime_titles = animes_df['title'].tolist()
    
    # Fix: Search bar is now empty by default (index=None)
    user_input = st.selectbox(
        "🔍 Search or select an anime to get started:", 
        options=all_anime_titles, 
        index=None,
        placeholder="Type or click to choose an anime..."
    )
            
    if st.button("🚀 Generate AI Recommendations", type="primary"):
        if user_input:
            if engine_choice == "CBF (Story DNA - Based on Plot & Genres)":
                recs, error_msg = get_cbf_recommendations(user_input, animes_df, tfidf_matrix, top_k_choice, selected_types, min_score)
            else:
                recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k_choice, selected_types, min_score)
                
            if error_msg:
                st.error(error_msg) 
            else:
                st.success("✅ Results generated successfully! (Low-rated items filtered out based on your settings)")
                
                # Added: Show the Target Anime DNA (Explainability)
                target_anime = animes_df[animes_df['title'] == user_input].iloc[0]
                raw_target_tags = str(target_anime['genres_detailed'])
                try:
                    target_tags = ast.literal_eval(raw_target_tags)
                except:
                    target_tags = raw_target_tags.replace("['", "").replace("']", "").split("', '")
                target_clean_tags = " • ".join([t.title() for t in target_tags if t])
                
                st.info(f"🎯 **Target Selected:** **{target_anime['title']}** (Type: {target_anime['type']} | Score: ⭐ {target_anime['score']:.2f})\n\n🏷️ **Story DNA:** {target_clean_tags}")
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
                            raw_score = float(row[sim_col])
                            match_pct = int((raw_score / max_sim) * 99) if max_sim > 0 else 0
                            
                            if match_pct >= 90:
                                stars = "★★★★★"
                                level_text = "Perfect Match"
                                star_color = "#FFD700"  
                            elif match_pct >= 75:
                                stars = "★★★★☆"
                                level_text = "Highly Similar"
                                star_color = "#F39C12"  
                            else:
                                stars = "★★★☆☆"
                                level_text = "Style Correlated"
                                star_color = "#AAB7B8"  
                                
                            st.markdown(f"""
                                <div style='text-align: right; padding-top: 12px;'>
                                    <div style='font-size: 20px; color: {star_color}; letter-spacing: 2px;'>{stars}</div>
                                    <div style='font-size: 13px; color: #888; margin-top: 4px; font-weight: 500;'>{level_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select or type an anime name first!")

# ---------------- Tab 2: Trending Leaderboard ----------------
with tab_trending:
    st.subheader("🏆 Community's Most Popular Anime Top 10")
    st.markdown("This list is extracted based on **rating frequency data** from real platform users (filtering out niche noise).")
    
    if not top10_df.empty:
        for rank, (index, row) in enumerate(top10_df.iterrows()):
            with st.container(border=True):
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
