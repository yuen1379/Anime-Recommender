import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 网页全局配置
# ==========================================
st.set_page_config(page_title="动漫双引擎推荐系统", page_icon="🎬", layout="wide")
st.title("🎬 动漫双引擎推荐系统")
st.markdown("基于 **内容标签 (CBF)** 与 **用户社区共鸣 (CF)** 的混合推荐引擎")
st.divider()

# ==========================================
# 2. 核心数据加载与缓存 (防止网页每次刷新都重新计算)
# ==========================================
@st.cache_data
def load_and_compute_models():
    # 1. 加载动漫基础数据
    animes_df = pd.read_csv("animes.csv")
    animes_df['genres_detailed'] = animes_df['genres_detailed'].fillna('')
    
    # 2. 构建 CBF 引擎 (TF-IDF 矩阵)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(animes_df['genres_detailed'])
    
    # 3. 构建 CF 引擎 (协同过滤矩阵)
    try:
        rating_df = pd.read_csv("rating_cf_ultra_final.csv")
        min_user_ratings = 20
        min_anime_ratings = 50
        
        active_users = rating_df['userID'].value_counts()
        active_users = active_users[active_users >= min_user_ratings].index
        filtered_ratings = rating_df[rating_df['userID'].isin(active_users)]
        
        active_animes = filtered_ratings['animeID'].value_counts()
        active_animes = active_animes[active_animes >= min_anime_ratings].index
        filtered_ratings = filtered_ratings[filtered_ratings['animeID'].isin(active_animes)]
        
        pivot_matrix = filtered_ratings.pivot_table(index='animeID', columns='userID', values='rating').fillna(0)
        item_similarity = cosine_similarity(pivot_matrix)
        item_sim_df = pd.DataFrame(item_similarity, index=pivot_matrix.index, columns=pivot_matrix.index)
    except Exception as e:
        st.error(f"⚠️ 协同过滤评分文件加载失败，请确保 'rating_cf_ultra_final.csv' 在同一目录下。错误: {e}")
        item_sim_df = pd.DataFrame()
        
    return animes_df, tfidf_matrix, item_sim_df

# 页面加载提示
with st.spinner("🤖 正在加载 AI 推荐引擎，首次启动需要几秒钟..."):
    animes_df, tfidf_matrix, item_sim_df = load_and_compute_models()

# ==========================================
# 3. 定义推荐函数 (直接照搬你 Jupyter 里的优秀代码)
# ==========================================
def get_cbf_recommendations(anime_title, df, feature_matrix, top_k=10):
    idx_list = df.index[df['title'].str.lower() == anime_title.lower()].tolist()
    if not idx_list:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    idx = idx_list[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    similar_indices = sim_scores.argsort()[-(top_k+1):][::-1][1:]
    recommendations = df.iloc[similar_indices][['title', 'type', 'score', 'genres_detailed']].copy()
    recommendations['similarity_score'] = sim_scores[similar_indices]
    return recommendations, None

def get_cf_recommendations(anime_title, df, sim_df, top_k=10):
    # 先把名字转换成 ID
    match = df[df['title'].str.lower() == anime_title.lower()]
    if match.empty:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    target_anime_id = match.iloc[0]['animeID']
    
    if target_anime_id not in sim_df.index:
        return None, f"🧊 【冷启动拦截】该动漫 (ID:{target_anime_id}) 打分人数过少，CF 引擎无法运算！请切换至 CBF 引擎。"
    
    sim_scores = sim_df[target_anime_id]
    similar_anime_ids = sim_scores.sort_values(ascending=False).index[1:top_k+1]
    
    results = []
    for aid in similar_anime_ids:
        match_anime = df[df['animeID'] == aid]
        if not match_anime.empty:
            row = match_anime.iloc[0].copy()
            row['cf_similarity'] = sim_scores[aid]
            results.append(row)
    return pd.DataFrame(results)[['title', 'type', 'score', 'genres_detailed', 'cf_similarity']], None

# ==========================================
# 4. 前端 UI 交互设计
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("🔍 请输入一部你喜欢的动漫 (如: Death Note, Toradora!, s-CRY-ed):", value="Death Note")

with col2:
    engine_choice = st.radio("⚙️ 请选择推荐引擎:", ["CF 协同过滤 (懂人心)", "CBF 内容推荐 (懂标签)"])

if st.button("🚀 生成专属推荐", type="primary"):
    if user_input:
        if engine_choice == "CBF 内容推荐 (懂标签)":
            recs, error_msg = get_cbf_recommendations(user_input, animes_df, tfidf_matrix, top_k=10)
        else:
            recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k=10)
            
        # 展示结果或报错
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"为您找到与《{user_input}》最相似的 10 部动漫：")
            st.dataframe(recs, use_container_width=True)
    else:
        st.info("请输入一部动漫的名字哦！")