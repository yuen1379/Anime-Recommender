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
        # ==========================================
        # 结果展示区 (Netflix 卡片式高级 UI)
        # ==========================================
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"✅ 成功为您找到与《{user_input}》最相似的 {top_k_choice} 部动漫：")
            st.markdown("---") # 加一条分割线
            
            # 动态判断当前用的是哪种相似度分数
            sim_col = 'similarity_score' if 'similarity_score' in recs.columns else 'cf_similarity'
            
            # 遍历推荐结果，一张一张画卡片
            for index, row in recs.iterrows():
                # 使用 container(border=True) 制造卡片效果
                with st.container(border=True):
                    # 把卡片分成左右两列 (左边占 3 份，右边占 1 份)
                    col_info, col_score = st.columns([3, 1])
                    
                    with col_info:
                        # 动漫标题 (加大加粗)
                        st.subheader(f"🎬 {row['title']}")
                        # 基础信息 (使用灰色小字)
                        st.caption(f"**类型**: {row['type']}  |  **大众评分**: ⭐ {row['score']}")
                        
                        # 把长长的一串微观标签藏在折叠面板里
                        # 优化后的高颜值“胶囊标签”面板
                        with st.expander("🏷️ 核心看点 / 剧情元素"):
                            import ast
                            
                            # 1. 安全解析标签数据
                            raw_tags = str(row['genres_detailed'])
                            try:
                                tags = ast.literal_eval(raw_tags)
                            except:
                                tags = raw_tags.replace("['", "").replace("']", "").split("', '")
                            
                            # 2. 限制展示数量，防止满屏密密麻麻影响视觉（最多展示前 10 个）
                            display_tags = tags[:10] if isinstance(tags, list) else []
                            
                            # 3. 使用 HTML+CSS 生成圆角胶囊样式 (适配深色/浅色模式的半透明背景)
                            badges_html = "".join([
                                f'<span style="display:inline-block; margin: 0px 6px 8px 0; padding: 4px 12px; '
                                f'background-color: rgba(130, 130, 130, 0.15); border: 1px solid rgba(130, 130, 130, 0.3); '
                                f'border-radius: 16px; font-size: 13px; white-space: nowrap;">{tag.title()}</span>'
                                for tag in display_tags if tag
                            ])
                            
                            # 4. 如果还有更多标签，用优雅的 "+X" 提示
                            if len(tags) > 10:
                                badges_html += f'<span style="display:inline-block; margin: 0px 6px 8px 0; padding: 4px 0px; font-size: 13px; color: gray;">+{len(tags)-10} 更多...</span>'
                                
                            # 渲染出自定义的 HTML UI
                            st.markdown(badges_html, unsafe_allow_html=True)
                            
                    with col_score:
                        # 计算匹配度百分比
                        match_pct = float(row[sim_col]) * 100
                        # 绘制数据指标和大数字
                        st.metric(label="✨ AI 匹配度", value=f"{match_pct:.1f}%")
                        # 绘制视觉进度条 (数值需在0.0到1.0之间，若出现极其罕见的>1截断处理)
                        st.progress(min(float(row[sim_col]), 1.0))
    else:
        st.info("请输入一部动漫的名字哦！")
