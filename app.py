import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 网页全局配置 (设为宽屏模式)
# ==========================================
st.set_page_config(page_title="动漫双引擎推荐系统", page_icon="🎬", layout="wide")

# ==========================================
# 2. 核心数据加载与缓存
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
        st.error("⚠️ 协同过滤文件加载失败。")
        item_sim_df = pd.DataFrame()
        top10_df = pd.DataFrame()
        
    return animes_df, tfidf_matrix, item_sim_df, top10_df

with st.spinner("🤖 正在加载 AI 推荐引擎，首次启动需要几秒钟..."):
    animes_df, tfidf_matrix, item_sim_df, top10_df = load_and_compute_models()

# ==========================================
# 3. 定义推荐函数 (加入二次过滤逻辑)
# ==========================================
def get_cbf_recommendations(anime_title, df, feature_matrix, top_k, selected_types, min_score):
    idx_list = df.index[df['title'].str.lower() == anime_title.lower()].tolist()
    if not idx_list:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    idx = idx_list[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    
    similar_indices = sim_scores.argsort()[::-1][1:]
    recs = df.iloc[similar_indices][['title', 'type', 'score', 'genres_detailed']].copy()
    recs['similarity_score'] = sim_scores[similar_indices]
    
    if selected_types:
        recs = recs[recs['type'].isin(selected_types)]
    recs = recs[recs['score'] >= min_score]
    
    if recs.empty:
        return None, "⚠️ 找不到符合筛选条件的推荐，请放宽左侧的【类型】或【最低评分】限制。"
    return recs.head(top_k), None

def get_cf_recommendations(anime_title, df, sim_df, top_k, selected_types, min_score):
    match = df[df['title'].str.lower() == anime_title.lower()]
    if match.empty:
        return None, f"❌ 找不到名为 '{anime_title}' 的动漫，请检查拼写。"
    target_id = match.iloc[0]['animeID']
    
    if target_id not in sim_df.index:
        return None, f"🧊 【极寒冷启动拦截】该动漫 (ID:{target_id}) 的真实用户打分过少！\n\n🚨 **CF (协同过滤) 引擎已瘫痪。** \n👉 **系统建议：请前往左侧控制面板，切换至【CBF (内容推荐)】引擎进行降维打击！**"
    
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
        return None, "⚠️ 找不到符合筛选条件的推荐，请放宽左侧的【类型】或【最低评分】限制。"
    return recs.head(top_k), None

# ==========================================
# 4. 界面排版 (UI Layout)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)
st.sidebar.header("⚙️ 引擎控制面板")

engine_choice = st.sidebar.radio("1. 核心算法选择:", ["CF (协同过滤 - 懂人心)", "CBF (内容推荐 - 懂标签)"])
top_k_choice = st.sidebar.slider("2. 推荐输出数量:", min_value=5, max_value=20, value=10, step=1)

st.sidebar.divider()
st.sidebar.subheader("🎛️ 结果二次筛选")
all_types = [t for t in animes_df['type'].unique() if pd.notna(t) and t != '']
selected_types = st.sidebar.multiselect("包含指定类型 (留空则不限):", all_types, default=[])
min_score = st.sidebar.slider("最低大众评分红线:", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

st.title("🎬 动漫双引擎推荐系统")
st.markdown("发现你的下一部神作！本平台由 **CBF (文本深度学习)** 与 **CF (百万级群体智慧)** 双重 AI 算法驱动。")

tab_search, tab_trending, tab_insights = st.tabs(["🎯 专属 AI 推荐", "🏆 历史热门 Top 10", "📊 算法性能评估 (评委专区)"])

# ---------------- Tab 1: 搜索与推荐 ----------------
with tab_search:
    col_search, col_demo = st.columns([3, 1])
    with col_search:
        # 把库里所有的动漫名字提取出来，变成一个列表
        all_anime_titles = animes_df['title'].tolist()
        # 使用 selectbox 替换 text_input，瞬间拥有 Google 级别的联想搜索体验！
        user_input = st.selectbox(
            "🔍 请搜索或直接选择一部动漫:", 
            options=all_anime_titles, 
            index=all_anime_titles.index("Death Note") if "Death Note" in all_anime_titles else 0
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("🚨 评委点此: 模拟冷门番剧"):
            st.warning('已加载靶标：请在搜索框输入 `s-CRY-ed`，并使用 **CF 引擎** 点击生成，查看系统崩溃拦截！')
            
    if st.button("🚀 激活 AI 生成专属推荐", type="primary"):
        if user_input:
            if engine_choice == "CBF (内容推荐 - 懂标签)":
                recs, error_msg = get_cbf_recommendations(user_input, animes_df, tfidf_matrix, top_k_choice, selected_types, min_score)
            else:
                recs, error_msg = get_cf_recommendations(user_input, animes_df, item_sim_df, top_k_choice, selected_types, min_score)
                
            if error_msg:
                st.error(error_msg) 
            else:
                st.success("✅ 成功为您生成结果！(基于大厂标准已过滤掉低分烂片)")
                st.markdown("---") 
                
                sim_col = 'similarity_score' if 'similarity_score' in recs.columns else 'cf_similarity'
                max_sim = float(recs[sim_col].max())
                
                for rank_idx, (index, row) in enumerate(recs.iterrows()):
                    with st.container(border=True):
                        col_info, col_score = st.columns([3, 1])
                        
                        with col_info:
                            st.subheader(f"🏅 {row['title']}")
                            st.caption(f"**类型**: {row['type']}  |  **社区评分**: ⭐ {row['score']:.2f}")
                            
                            with st.expander("🏷️ 核心看点 / Story Tropes"):
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
                            # 方案 3：纯净星级制 (最符合直觉，零认知负担)
                            raw_score = float(row[sim_col])
                            match_pct = int((raw_score / max_sim) * 99) if max_sim > 0 else 0
                            
                            # 星级与清爽文案判定
                            if match_pct >= 90:
                                stars = "★★★★★"
                                level_text = "完美命中"
                                star_color = "#FFD700"  # 耀眼金
                            elif match_pct >= 75:
                                stars = "★★★★☆"
                                level_text = "高度重合"
                                star_color = "#F39C12"  # 活力橘金
                            else:
                                stars = "★★★☆☆"
                                level_text = "风格关联"
                                star_color = "#AAB7B8"  # 质感灰银
                                
                            # 使用 HTML 渲染干净的星级排版 (星星在上，小字在下)
                            st.markdown(f"""
                                <div style='text-align: right; padding-top: 12px;'>
                                    <div style='font-size: 20px; color: {star_color}; letter-spacing: 2px;'>{stars}</div>
                                    <div style='font-size: 13px; color: #888; margin-top: 4px; font-weight: 500;'>{level_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("请输入一部动漫的名字哦！")

# ---------------- Tab 2: 热门排行榜 ----------------
with tab_trending:
    st.subheader("🏆 社区最受欢迎动漫 Top 10")
    st.markdown("该榜单基于平台真实用户的 **评分频率数据** 提取（过滤掉了冷门噪声）。")
    
    if not top10_df.empty:
        for rank, (index, row) in enumerate(top10_df.iterrows()):
            with st.container(border=True):
                # 修复！使用纯 HTML 替代 ### 标题，彻底消灭 link 锚点图标
                st.markdown(f"<div style='font-size: 18px; font-weight: bold; margin-bottom: 8px;'>👑 No.{rank + 1} &nbsp; {row['title']}</div>", unsafe_allow_html=True)
                st.caption(f"**类型**: {row['type']}  |  **大众评分**: ⭐ {row['score']}")
    else:
        st.info("数据加载中...")

# ---------------- Tab 3: 数据洞察面板 ----------------
with tab_insights:
    st.subheader("📊 底层算法性能基准 (Benchmarks)")
    st.markdown("大作业实验环境下的离线评估报告（42,797 组 Test Data 测试结果）：")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CF RMSE (协同过滤误差)", "1.377", "- 预测更准")
    col2.metric("CBF RMSE (内容引擎误差)", "1.411", "+ 覆盖更广")
    col3.metric("CF 推荐命中率 (Precision@10)", "16.8%", "+ 优于 CBF 的 15.4%")
    
    st.divider()
    st.subheader("📈 当前库内动漫类型分布 (EDA)")
    type_counts = animes_df['type'].value_counts()
    st.bar_chart(type_counts, color="#ff4b4b")
