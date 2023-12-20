import plotly.graph_objects as go
import pandas as pd

# 示例数据（替换为你的实际数据）
df_medals = pd.DataFrame({
    'Country': ['风电', '光伏', '系统电输入', '系统气输入', 'Russia'],
    'Gold Medals': [39, 38, 27, 22, 20],
    'Silver Medals': [41, 32, 14, 21, 28],
    'Bronze Medals': [33, 18, 17, 22, 23]
})

NUM_COUNTRIES = len(df_medals)
X_POS, Y_POS = 0.5, 1 / (NUM_COUNTRIES - 1)
NODE_COLORS = ["seagreen", "dodgerblue", "orange", "palevioletred", "darkcyan"]
LINK_COLORS = ["lightgreen", "lightskyblue", "bisque", "pink", "lightcyan"]

source = []
node_x_pos, node_y_pos = [], []
node_labels, node_colors = [], NODE_COLORS[0:NUM_COUNTRIES]
link_labels, link_colors, link_values = [], [], []

# 节点和链接的设置（与示例数据相关）
for i in range(NUM_COUNTRIES):
    source.extend([i] * 3)
    node_x_pos.append(0.01)
    node_y_pos.append(round(i * Y_POS + 0.01, 2))
    country = df_medals['Country'][i]
    node_labels.append(country)
    for medal in ["Gold", "Silver", "Bronze"]:
        link_labels.append(f"{country}-{medal}")
        link_values.append(df_medals[f"{medal} Medals"][i])
    link_colors.extend([LINK_COLORS[i]] * 3)

source_last = max(source) + 1
target = [source_last, source_last + 1, source_last + 2] * NUM_COUNTRIES
target_last = max(target) + 1

node_labels.extend(["Gold", "Silver", "Bronze"])
node_colors.extend(["gold", "silver", "brown"])
node_x_pos.extend([X_POS, X_POS, X_POS])
node_y_pos.extend([0.01, 0.5, 1])

# 最后一组链接和节点
source.extend([source_last, source_last + 1, source_last + 2])
target.extend([target_last] * 3)
node_labels.extend(["Total Medals"])
node_colors.extend(["grey"])
node_x_pos.extend([X_POS + 0.25])
node_y_pos.extend([0.5])

for medal in ["Gold", "Silver", "Bronze"]:
    link_labels.append(f"{medal}")
    link_values.append(df_medals[f"{medal} Medals"][:i + 1].sum())
link_colors.extend(["gold", "silver", "brown"])

# 创建桑基图的节点和链接
NODES = dict(pad=20, thickness=20,
             line=dict(color="lightslategrey", width=0.5),
             hovertemplate=" ",
             label=node_labels,
             color=node_colors,
             x=node_x_pos,
             y=node_y_pos)
LINKS = dict(source=source,
             target=target,
             value=link_values,
             label=link_labels,
             color=link_colors,
             hovertemplate="%{label}")

# 创建桑基图的图表
data = go.Sankey(arrangement='snap',
                 node=NODES,
                 link=LINKS)

# 更新图表属性
fig = go.Figure(data)
fig.update_traces(valueformat='3d',
                  valuesuffix=' Medals',
                  selector=dict(type='sankey'))
fig.update_layout(title="Olympics - 2021: Country & Medals",
                  font_size=16,
                  width=1200,
                  height=500)
fig.update_layout(hoverlabel=dict(bgcolor="grey",
                                  font_size=14,
                                  font_family="Rockwell"))

# 显示图表
fig.show()
