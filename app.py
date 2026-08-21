import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 网页全局配置 (设为宽屏模式)
# ==========================================
st.set_page_config(page_title="动漫双引擎推荐系统", page_icon="🎬", layout="wide")

# ==========================================
# 2. 核心数据加载与缓存 (核心算法保持不变)
# ==========================================
@st.cache_data
def load_and_compute_models():
    animes_df = pd.read_csv("animes.csv")
    animes_df['genres_detailed'] = animes_df['genres_detailed'].fillna('')
    
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
        
        pivot_matrix = filtered_ratings.pivot_table(index='animeID', columns='userID', values='rating').fillna(0)
        item_sim_df = pd.DataFrame(cosine_similarity(pivot_matrix), index=pivot_matrix.index, columns=pivot_matrix.index)
    except Exception as e:
        st.error("⚠️ 协同过滤文件加载失败。")
        item_sim_df = pd.DataFrame()
        
    return animes_df, tfidf_matrix, item_sim_df

with st.spinner("🤖 正在加载 AI 推荐引擎，请稍候..."):
    animes_df, tfidf_matrix, item_sim_df = load_and_compute_models()

# ==========================================
# 3. 定义推荐函数 (保持不变)
# ==========================================
def get_cbf_recommendations(anime_title, df, feature_matrix, top_k):
    idx_list = df.index[df['title'].str.lower() == anime_title.lower()].tolist()
    if not idx_list:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    idx = idx_list[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    similar_indices = sim_scores.argsort()[-(top_k+1):][::-1][1:]
    recs = df.iloc[similar_indices][['title', 'type', 'score', 'genres_detailed']].copy()
    recs['similarity_score'] = sim_scores[similar_indices]
    return recs, None

def get_cf_recommendations(anime_title, df, sim_df, top_k):
    match = df[df['title'].str.lower() == anime_title.lower()]
    if match.empty:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    target_id = match.iloc[0]['animeID']
    
    if target_id not in sim_df.index:
        return None, f"🧊 【冷启动拦截】该动漫 (ID:{target_id}) 打分人数过少，CF 引擎无法运算！请切换至 CBF 引擎。"
    
    sim_scores = sim_df[target_id]
    similar_ids = sim_scores.sort_values(ascending=False).index[1:top_k+1]
    
    results = []
    for aid in similar_ids:
        match_anime = df[df['animeID'] == aid]
        if not match_anime.empty:
            row = match_anime.iloc[0].copy()
            row['cf_similarity'] = sim_scores[aid]
            results.append(row)
    return pd.DataFrame(results)[['title', 'type', 'score', 'genres_detailed', 'cf_similarity']], None

# ==========================================
# 4. 界面排版 (UI Layout)
# ==========================================
# --- 左侧边栏 (Sidebar) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100) # 放个小图标增加专业感
st.sidebar.header("⚙️ 控制面板")

engine_choice = st.sidebar.radio(
    "选择推荐引擎:", 
    ["CF (协同过滤 - 懂人心)", "CBF (内容推荐 - 懂标签)"]
)

# 新增：让用户自己滑块选择想看几个推荐 (交互感拉满！)
top_k_choice = st.sidebar.slider("推荐数量 (Top-K):", min_value=5, max_value=20, value=10, step=1)

st.sidebar.divider()
st.sidebar.info(
    "💡 **引擎使用指南**\n\n"
    "- **CF (协同过滤)**: 基于大众真实打分，适合寻找跨类型的高分神作。\n"
    "- **CBF (内容推荐)**: 基于微观剧情标签，适合寻找同IP或无评分的新番。"
)

# --- 主体内容区 (Main Area) ---
st.title("🎬 动漫双引擎推荐系统")
st.markdown("输入你喜欢的动漫，AI 将根据你选择的底层算法，为你寻找下一部神作！")

# 搜索框居中放宽
user_input = st.text_input("🔍 请输入动漫名称 (如: Death Note, Toradora!, s-CRY-ed):", value="Death Note")

if st.button("🚀 生成专属推荐", type="primary"):
    if user_input:
        if engine_choice == "CBF (内容推荐 - 懂标签)":
            recs, error_msg = get_cbf_recommendations(user_input, animes_df, tfidf_matrix, top_k=top_k_choice)
        else:
            recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k=top_k_choice)
            
        # 结果展示区
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"✅ 成功为您找到与《{user_input}》最相似的 {top_k_choice} 部动漫：")
            # 使用更加清爽的数据表展示方式
            st.dataframe(recs, use_container_width=True, hide_index=True)
    else:
        st.info("请输入一部动漫的名字哦！")
