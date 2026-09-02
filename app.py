import streamlit as st
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote

hide_ui_style = """
<style>
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(hide_ui_style, unsafe_allow_html=True)


st.set_page_config(page_title="Anime Dual-Engine Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data
def load_and_compute_models():
    animes_df = pd.read_csv("anime_safe.csv")                     
    animes_df['genre'] = animes_df['genre'].fillna('')         
    animes_df['type'] = animes_df['type'].fillna('Unknown')
    animes_df['rating'] = pd.to_numeric(animes_df['rating'], errors='coerce').fillna(6.0)  # ← score -> rating


    def clean_tags(tag_str):
        if not tag_str:
            return []
        return [t.strip().title() for t in tag_str.split(',') if t.strip()]

    animes_df['clean_tags_list'] = animes_df['genre'].apply(clean_tags)   # ← genres_detailed -> genre

    # 2. CBF 引擎：多模态特征融合
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(animes_df['genre'])        # ← genres_detailed -> genre

    type_matrix = sp.csr_matrix(pd.get_dummies(animes_df['type']).values)
    score_matrix = sp.csr_matrix(MinMaxScaler().fit_transform(animes_df[['rating']]))  # ← score -> rating

    cbf_feature_matrix = sp.hstack([tfidf_matrix * 1.0, type_matrix * 0.5, score_matrix * 0.5])

    # 3. CF 引擎
    cf_status = "OK"
    try:
        rating_df = pd.read_csv("rating_safe.csv")             
        active_users = rating_df['user_id'].value_counts()        # ← userID -> user_id
        active_users = active_users[active_users >= 20].index
        filtered_ratings = rating_df[rating_df['user_id'].isin(active_users)]

        active_animes = filtered_ratings['anime_id'].value_counts()  # ← animeID -> anime_id
        active_animes = active_animes[active_animes >= 50].index
        filtered_ratings = filtered_ratings[filtered_ratings['anime_id'].isin(active_animes)]

        top_anime_ids = filtered_ratings['anime_id'].value_counts().head(10).index
        top10_df = animes_df.set_index('anime_id').loc[top_anime_ids].reset_index()  # ← animeID -> anime_id

        pivot_matrix = filtered_ratings.pivot_table(index='anime_id', columns='user_id', values='rating')
        user_mean = pivot_matrix.mean(axis=0)
        pivot_matrix_centered = pivot_matrix.sub(user_mean, axis=1).fillna(0)

        item_sim_df = pd.DataFrame(cosine_similarity(pivot_matrix_centered), index=pivot_matrix.index, columns=pivot_matrix.index)
    except Exception as e:
        cf_status = f"ERROR: {str(e)}"
        item_sim_df = pd.DataFrame()
        top10_df = pd.DataFrame()

    return animes_df, cbf_feature_matrix, item_sim_df, top10_df, cf_status

with st.spinner("🤖 Loading Upgraded AI Engine (Multimodal Stacking & Mean-Centering)..."):
    animes_df, cbf_feature_matrix, item_sim_df, top10_df, cf_status = load_and_compute_models()

def get_cbf_recommendations(anime_title, df, feature_matrix, top_k, selected_types, min_score, max_episodes=0):
    idx_list = df.index[df['name'].str.lower() == anime_title.lower()].tolist()  # ← title -> name
    if not idx_list:
        return None, f"❌ Cannot find an anime named '{anime_title}'. Please check your spelling."
    idx = idx_list[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()

    similar_indices = sim_scores.argsort()[::-1][1:]
    recs = df.iloc[similar_indices][['name', 'type', 'rating', 'genre', 'episodes', 'members', 'clean_tags_list']].copy()
    recs['similarity_score'] = sim_scores[similar_indices]

    if selected_types:
        recs = recs[recs['type'].isin(selected_types)]
    recs = recs[recs['rating'] >= min_score]
    if max_episodes > 0:
        recs = recs[(recs['episodes'] <= max_episodes) | (recs['episodes'].isna())]
        
    if recs.empty:
        return None, "⚠️ No recommendations match your filters. Please relax the 'Type' or 'Minimum Rating' limits on the left."
    return recs.head(top_k), None

DARK_GENRE_FLAGS = ["Horror", "Psychological", "Gore", "Thriller"]

def get_content_note(genre_str):
    if not genre_str:
        return None
    flags = [g for g in DARK_GENRE_FLAGS if g.lower() in genre_str.lower()]
    if flags:
        return f"⚠️ Contains {', '.join(flags)} themes — heads up if that's not your thing."
    return None

def humanize_members(members):
    try:
        m = float(members)
    except (TypeError, ValueError):
        return None
    if m >= 1_000_000:
        return f"❤️ Loved by {m/1_000_000:.1f}M+ fans"
    elif m >= 1000:
        return f"❤️ Loved by {int(m/1000)}K+ fans"
    else:
        return f"❤️ Loved by {int(m)} fans"

def get_rating_percentile(rating_value, all_ratings):
    if pd.isna(rating_value):
        return None
    percentile = (all_ratings < rating_value).mean() * 100
    return round(percentile)

def genre_overlap_percentage(target_genre_str, rec_genre_str):
    if not target_genre_str or not rec_genre_str:
        return 0
    target_set = set(g.strip().lower() for g in target_genre_str.split(',') if g.strip())
    rec_set = set(g.strip().lower() for g in rec_genre_str.split(',') if g.strip())
    if not target_set:
        return 0
    overlap = target_set & rec_set
    return round(len(overlap) / len(target_set) * 100)

def get_cf_recommendations(anime_title, df, sim_df, top_k, selected_types, min_score, max_episodes=0):
    match = df[df['name'].str.lower() == anime_title.lower()]
    if match.empty:
        return None, f"❌ Cannot find an anime named '{anime_title}'. Please check your spelling."
    target_id = match.iloc[0]['anime_id']

    if target_id not in sim_df.index:
        return None, f"🧊 [Cold Start Intercept] This anime doesn't have enough community ratings yet!\n\n🚨 **CF Engine cannot process this.** \n👉 **Suggestion: Switch to the [CBF (Story DNA)] engine on the left panel to analyze it by plot instead!**"

    sim_scores = sim_df[target_id]
    similar_ids = sim_scores.sort_values(ascending=False).index[1:]

    results = []
    for aid in similar_ids:
        match_anime = df[df['anime_id'] == aid]
        if not match_anime.empty:
            row = match_anime.iloc[0].copy()
            row['cf_similarity'] = sim_scores[aid]
            results.append(row)

    recs = pd.DataFrame(results)[['name', 'type', 'rating', 'genre', 'episodes', 'members', 'clean_tags_list', 'cf_similarity']]

    if selected_types:
        recs = recs[recs['type'].isin(selected_types)]
    recs = recs[recs['rating'] >= min_score]
    if max_episodes > 0:
        recs = recs[(recs['episodes'] <= max_episodes) | (recs['episodes'].isna())]

    if recs.empty:
        return None, "⚠️ No recommendations match your filters. Please relax the 'Type' or 'Minimum Rating' limits on the left."
    return recs.head(top_k), None

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)
st.sidebar.header("⚙️ Engine Control Panel")

if 'surprise_pick' not in st.session_state:
    st.session_state.surprise_pick = None

if st.sidebar.button("🎲 Surprise Me!", width="stretch"):
    st.session_state.surprise_pick = animes_df.sample(1)['name'].values[0]
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

max_episodes = st.sidebar.slider("Max Episodes (0 = no limit):", min_value=0, max_value=500, value=0, step=10)
st.sidebar.caption("💡 Leave 'Include Specific Types' blank to include all types, or select specific ones to narrow down.")
st.sidebar.caption("🔍 Filters apply *after* generating recommendations — narrowing too much may return fewer results.")

st.title("🎬 Anime Dual-Engine Recommendation System")
st.markdown("Discover your next masterpiece! Powered by dual AI algorithms analyzing both **Story DNA** and **Community Wisdom**.")

tab_search, tab_trending, tab_insights = st.tabs(["🎯 Exclusive AI Recommendations", "🏆 All-Time Top 10 Trending", "📊 Algorithm Benchmarks"])

with tab_search:
    all_anime_titles = animes_df['name'].tolist()                              # ← title -> name

    default_index = None
    if st.session_state.surprise_pick and st.session_state.surprise_pick in all_anime_titles:
        default_index = all_anime_titles.index(st.session_state.surprise_pick)

    user_input = st.selectbox(
        "🔍 Search or select an anime to get started:",
        options=all_anime_titles,
        index=default_index,
        placeholder="Type or click to choose an anime..."
    )

    if st.button("🚀 Generate AI Recommendations", type="primary"):
        if user_input:
            if engine_choice == "CBF (Story DNA - Based on Plot & Genres)":
                recs, error_msg = get_cbf_recommendations(user_input, animes_df, cbf_feature_matrix, top_k_choice, selected_types, min_score, max_episodes)
            else:
                if cf_status != "OK":
                    recs, error_msg = None, f"🚨 System Error: CF Engine failed to load ({cf_status}). Please check data files."
                else:
                    recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k_choice, selected_types, min_score, max_episodes)

            if error_msg:
                st.error(error_msg)
            else:
                st.success("✅ Results generated successfully! (Low-rated items filtered out based on your settings)")

                target_anime = animes_df[animes_df['name'] == user_input].iloc[0]  # ← title -> name
                target_clean_tags = " • ".join(target_anime['clean_tags_list'])
                st.info(f"🎯 **Target Selected:** **{target_anime['name']}** (Type: {target_anime['type']} | Score: ⭐ {target_anime['rating']:.2f})\n\n🏷️ **Story DNA:** {target_clean_tags}")
                st.markdown("---")

                sim_col = 'similarity_score' if 'similarity_score' in recs.columns else 'cf_similarity'

                for rank_idx, (index, row) in enumerate(recs.iterrows()):
                    with st.container(border=True):
                        col_info, col_score = st.columns([3, 1])

                        with col_info:
                            st.subheader(f"🏅 {row['name']}")
                            st.caption(f"**Type**: {row['type']}  |  **Community Rating**: ⭐ {row['rating']:.2f}")

                            percentile = get_rating_percentile(row['rating'], animes_df['rating'])
                            if percentile is not None:
                                st.caption(f"📊 Rated higher than {percentile}% of anime in our database")
                            
                            if sim_col == 'similarity_score':
                                reason = f"Because it shares similar genres and format with **{target_anime['name']}**"
                            else:
                                reason = f"Because fans of **{target_anime['name']}** also tend to enjoy this one"
                            st.caption(f"💡 {reason}")

                            episodes_val = row.get('episodes')
                            if pd.notna(episodes_val) and str(episodes_val).replace('.', '', 1).isdigit() and float(episodes_val) > 0:
                                episodes_text = f"{int(float(episodes_val))} episodes"
                            else:
                                episodes_text = "Episode count unknown"
                            st.caption(f"📺 {episodes_text}")

                            content_note = get_content_note(row.get('genre', ''))
                            if content_note:
                                st.caption(content_note)
                                
                            fan_text = humanize_members(row.get('members'))
                            if fan_text:
                                st.caption(fan_text)

                            overlap_pct = genre_overlap_percentage(target_anime['genre'], row.get('genre', ''))
                            st.caption(f"🧬 Genre Overlap: {overlap_pct}%")
                            st.progress(overlap_pct / 100)
                            
                            crunchyroll_url = f"https://www.crunchyroll.com/search?q={quote(row['name'])}"
                            st.link_button("📺 Search on Crunchyroll", crunchyroll_url, width="stretch")
                            
                            search_query = quote(f"{row['name']} anime")
                            mal_url = f"https://myanimelist.net/anime.php?q={search_query}"
                            google_url = f"https://www.google.com/search?q={search_query}"
                            
                            link_col1, link_col2 = st.columns(2)
                            with link_col1:
                                st.link_button("🔍 MyAnimeList", mal_url, width="stretch")
                            with link_col2:
                                st.link_button("🌐 Google", google_url, width="stretch")

                            
                            with st.expander("🏷️ Core Tropes / Tags"):
                                display_tags = row['clean_tags_list']
                                badges_html = "".join([
                                    f'<span style="display:inline-block; margin: 0px 6px 8px 0; padding: 4px 12px; '
                                    f'background-color: rgba(130, 130, 130, 0.15); border: 1px solid rgba(130, 130, 130, 0.3); '
                                    f'border-radius: 16px; font-size: 13px; white-space: nowrap;">{tag}</span>'
                                    for tag in display_tags
                                ])
                                st.markdown(badges_html, unsafe_allow_html=True)

                        with col_score:
                            rank_position = rank_idx 
                            total = len(recs)

                            if rank_position < total * 0.2:
                                stars, level_text, star_color = "★★★★★", "Perfect Match", "#FFD700"
                            elif rank_position < total * 0.5:
                                stars, level_text, star_color = "★★★★☆", "Highly Similar", "#F39C12"
                            else:
                                stars, level_text, star_color = "★★★☆☆", "Style Correlated", "#AAB7B8"

                            st.markdown(f"""
                                <div style='text-align: right; padding-top: 12px;'>
                                    <div style='font-size: 20px; color: {star_color}; letter-spacing: 2px;'>{stars}</div>
                                    <div style='font-size: 13px; color: #888; margin-top: 4px; font-weight: 500;'>{level_text}</div>
                                </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("📝 Help us improve (System Evaluation)")
                st.write("Does this AI-generated recommendation list match your expectations?")
                feedback = st.feedback("faces")
                if feedback is not None:
                    st.toast("Thank you for your feedback! This helps optimize our algorithm.", icon="🎉")
        else:
            st.warning("⚠️ Please select or type an anime name first!")

with tab_trending:
    st.subheader("🏆 Community's Most Popular Anime Top 10")
    st.markdown("This list is extracted based on **rating frequency data** from real platform users (filtering out niche noise).")

    if not top10_df.empty:
        for rank, (index, row) in enumerate(top10_df.iterrows()):
            with st.container(border=True):
                st.markdown(f"<div style='font-size: 18px; font-weight: bold; margin-bottom: 8px;'>👑 No.{rank + 1} &nbsp; {row['name']}</div>", unsafe_allow_html=True)  # ← title -> name
                st.caption(f"**Type**: {row['type']}  |  **Community Rating**: ⭐ {row['rating']}")  # ← score -> rating
    else:
        st.info("Loading data...")

with tab_insights:
    st.subheader("📊 Model Evaluation (Offline Benchmark)")
    st.markdown("""
    *Note: Predictive performance metrics are strictly evaluated offline via train/test splitting (**329,973 test ratings**). UI dynamically loads Top-K values.*
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CF RMSE", "1.130", "- Lower Error 🏆")
    col2.metric("CBF RMSE", "1.312", "+ Higher Error")
    col3.metric("CF Hit Rate (P@10)", "13.6%", "+ Better Relevance 🏆")
    col4.metric("CBF Hit Rate (P@10)", "7.1%", "- Lower Relevance")

    st.divider()

    st.subheader("📈 Detailed Performance Matrix")

    eval_data = {
        "Evaluation Metric": [
            "RMSE (Predictive Accuracy) ↓",
            "Accuracy (Threshold >= 7) ↑",
            "Precision@10 (Top-10 Hit Rate) ↑",
            "Recall@10 (Coverage Ability) ↑",
            "F1-Score@10 (Balance Score) ↑"
        ],
        "CF (Community Favorites)": [
            "1.130 (Winner 🏆)",
            "86.0% (Winner 🏆)",
            "13.6% (Winner 🏆)",
            "4.4% (Winner 🏆)",
            "0.067 (Winner 🏆)"
        ],
        "CBF (Story DNA)": [
            "1.312",
            "83.7%",
            "7.1%",
            "2.1%",
            "0.032"
        ],
        "Popularity Baseline": [
            "N/A",
            "N/A",
            "2.3%",
            "0.9%",
            "0.013"
        ]
    }
    eval_df = pd.DataFrame(eval_data)
    eval_df = eval_df[['Evaluation Metric', 'Popularity Baseline', 'CBF (Story DNA)', 'CF (Community Favorites)']]
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.info("""
    **🎓 Final Verdict: Which model is more outstanding?**

    * **Collaborative Filtering (CF)** swept all numerical metrics across the board (Accuracy, Precision, Recall, F1) because it effectively leverages actual human behavior and mitigates rating bias via Mean-Centering. It excels at predicting what users truly want, beating the generic Popularity Baseline by ~5.9x.
    * **Content-Based Filtering (CBF)** may score lower in raw offline metrics, but it fundamentally solves the **'Cold-Start Problem'**. It requires zero historical user data, making it computationally essential for newly released anime, and provides vital system stability when CF fails.

    **System Architecture Conclusion:** Neither model is perfect alone. The most robust architecture is a **Dual-Engine Fallback System**—using CF for highly-rated popular titles to maximize accuracy, and seamlessly falling back to stacked-metadata CBF when facing sparse or zero-data environments.
    """)
