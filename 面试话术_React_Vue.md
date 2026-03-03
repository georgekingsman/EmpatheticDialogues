# React / Vue 面试高频问题 & 回答话术（中英对照）

---

## Q1: "你在项目中用了 React 哪些核心概念？"

**EN:** "I heavily used React Hooks throughout the project — `useState` for managing sensor data and UI states, `useEffect` for data fetching with polling intervals and cleanup, and `useRef` for DOM references. I also used React Router for client-side routing with conditional route guards based on user roles — admin, active, or inactive."

**中文：** "我在项目中大量使用了 React Hooks — `useState` 管理传感器数据和 UI 状态，`useEffect` 实现数据轮询与清理，`useRef` 用于 DOM 引用。同时使用 React Router 实现客户端路由，并根据用户角色（管理员/已激活/未激活）做条件路由守卫。"

**可以直接指向的代码：**

```jsx
// App.js — 条件路由守卫（基于角色的权限控制）
<Routes>
  {(userData?.is_active) && (<Route path="/sensors" element={<Sensors />} />)}
  {(userData?.is_admin) && (<Route path="/edit" element={<EditSensor />} />)}
  {(!userData?.email) && (<Route path="/login" element={<Login />} />)}
  {(!userData?.is_active && userData?.email) && (
    <Route path="/inactivate" element={<InActive />} />
  )}
</Routes>
```

---

## Q2: "你怎么处理实时数据更新的？"

**EN:** "I implemented a polling mechanism using `setInterval` inside `useEffect`. The dashboard fetches sensor data every 5 seconds from the backend API. I also track each sensor's last updated timestamp so the UI can visually indicate stale data — if a sensor hasn't reported in over 10 minutes, the name turns grey and the status shows red. I made sure to return a cleanup function to clear the interval when the component unmounts."

**中文：** "我在 `useEffect` 里用 `setInterval` 实现轮询机制，仪表盘每 5 秒从后端 API 拉取传感器数据。同时追踪每个传感器的最后更新时间戳——如果超过 10 分钟没有上报，名称变灰、状态变红。组件卸载时通过返回的清理函数清除定时器，防止内存泄漏。"

**典型代码：**

```jsx
// Sensors.js — 实时轮询 + 清理
useEffect(() => {
  fetchLastRecord();
  const interval = setInterval(fetchLastRecord, 5000);
  return () => clearInterval(interval);  // 防止内存泄漏
}, []);

// SensorCard — 实时计时器显示"X秒前更新"
useEffect(() => {
  const interval = setInterval(() => {
    setTimeSinceUpdate(Math.floor((Date.now() - lastUpdated) / 1000));
  }, 1000);
  return () => clearInterval(interval);
}, [lastUpdated]);
```

---

## Q3: "你用了哪些性能优化手段？"

**EN:** "The sensor list can have 50+ devices, so I used `react-window`'s `FixedSizeList` for virtualized rendering — only visible rows are rendered in the DOM, which dramatically improves scroll performance on mobile. For the history chart, I implemented data downsampling — dividing timestamps into intervals based on the time period (1H/3H/24H) and averaging the values within each interval, rather than plotting every single data point."

**中文：** "传感器列表可能有 50 多个设备，所以我用了 `react-window` 的 `FixedSizeList` 做虚拟化渲染——只渲染可视区域内的行，大幅提升移动端滚动性能。对于历史图表，我实现了数据降采样——根据时间段（1H/3H/24H）将时间戳分组并取平均值，而不是绘制每一个数据点。"

**典型代码：**

```jsx
// Sensors.js — 虚拟化列表
<FixedSizeList
  height={375}
  itemSize={75}
  itemCount={filteredRecords.length}
  width="100%"
>
  {({ index, style }) => (
    <Row style={style} data={filteredRecords[index]}
         onClick={() => setSelectedSensorId(filteredRecords[index][0])} />
  )}
</FixedSizeList>
```

```jsx
// HistoryChart.js — 数据降采样（时间窗口内取平均值）
const interval = (dataPeriod === 1) ? hour1_interval 
               : (dataPeriod === 3) ? hour3_interval 
               : hour24_interval;
const filteredTimestamps = timestamps.filter((_, i) => i % interval === 0);
const values = filteredTimestamps.map((timestamp, _index) => {
  const _templist = timestamps.slice(previousIndex, originalIndex);
  const _values = _templist.map(ts => data[...][ts][...] / divisor);
  return (_values.reduce((a, b) => a + b, 0) / _templist.length).toFixed(2);
});
```

---

## Q4: "你怎么实现告警系统的？"

**EN:** "I defined threshold values for each sensor type directly in the component — for example, CO₂ above 1000 ppm, temperature outside 15-30°C, humidity outside 30-70%. I also calculated the humidity ratio using the IAPWS IF-97 thermodynamic formula to detect condensation risk. When any threshold is exceeded, the card border turns red, and individual values are highlighted. This gives operators an at-a-glance view of which sensors need attention."

**中文：** "我在组件中直接定义每种传感器的阈值——比如 CO₂ 超过 1000 ppm、温度在 15-30°C 范围外、湿度在 30-70% 范围外。还用 IAPWS IF-97 热力学公式计算了湿度比来检测凝结风险。当任何阈值被超过，卡片边框变红，异常值高亮显示，让操作人员一眼就能看到哪些传感器需要关注。"

**典型代码：**

```jsx
// Sensors.js — 多级告警判断
const scd_co2_alarm = (data.scd?.["3"] > 1000);
const scd_temp_alarm = (data.scd?.["4"]/100 > 30) || (data.scd?.["4"]/100 < 15);
const scd_humdity_alarm = (data.scd?.["5"]/100 > 70) || (data.scd?.["5"]/100 < 30);
const scd_alarm = (scd_co2_alarm || scd_temp_alarm || scd_humdity_alarm);

// 视觉反馈 — 动态边框颜色
<Card style={{ borderColor: scd_alarm ? '#990000' : '#009999' }}>
  <Typography style={{ color: scd_co2_alarm ? '#BB0000' : 'inherit' }}>
    {data.scd["3"]} ppm
  </Typography>
</Card>
```

---

## Q5: "为什么用 React 而不是 Vue？或者说你怎么看这两个框架的区别？"

**EN:** "I actually used both in this project ecosystem. React was our primary choice because of JSX flexibility — when building complex sensor dashboards with conditional rendering and dynamic data transformations, JSX gives you the full power of JavaScript inline. Vue was used in some earlier components for its simpler template syntax, which is faster for building standard CRUD pages. In practice, the two are very similar in capability — React gives more flexibility with its 'everything is JavaScript' philosophy, while Vue provides more structure out of the box with its Options API and built-in directives like `v-for` and `v-if`."

**中文：** "我在这个项目生态中实际上两个都用了。React 是主要选择，因为 JSX 的灵活性——在构建复杂的传感器仪表盘时需要大量条件渲染和动态数据转换，JSX 让你可以直接内联使用 JavaScript 的全部能力。Vue 用在了一些早期组件中，因为它的模板语法更简洁，构建标准 CRUD 页面更快。实际上两者能力非常相似——React 的 'everything is JavaScript' 哲学更灵活，Vue 则通过 Options API 和内置指令（如 `v-for`、`v-if`）提供更多开箱即用的结构。"

---

## Q6: "你状态管理用的什么方案？为什么没用 Redux？"

**EN:** "I used React's built-in `useState` and `useEffect` hooks for state management, without Redux or Context API. The reason is that our data flow is relatively straightforward — sensor data flows from one API endpoint down to child components via props. There's no complex cross-component state sharing that would justify the Redux boilerplate. The user authentication state is fetched once in `App.js` and passed down as props. For a larger team or more complex state interactions, I would consider Redux Toolkit or Zustand."

**中文：** "我使用 React 内置的 `useState` 和 `useEffect` 来管理状态，没有用 Redux 或 Context API。原因是我们的数据流相对直接——传感器数据从一个 API 端点流向子组件，通过 props 传递。没有复杂的跨组件状态共享来证明引入 Redux 样板代码的必要性。用户认证状态在 `App.js` 中获取一次，通过 props 向下传递。如果团队更大或状态交互更复杂，我会考虑 Redux Toolkit 或 Zustand。"

---

## Q7: "你项目的数据可视化是怎么做的？"

**EN:** "I used Chart.js via `react-chartjs-2` for time-series visualization. The history chart component supports switching between 1-hour, 3-hour, and 24-hour windows with a date picker. A key technical detail is the data downsampling — for a 24-hour view, I average values within time intervals to keep the chart responsive, rather than plotting thousands of raw data points. For the PM sensor and weather station pages, I used bar charts with similar aggregation logic."

**中文：** "我使用 `react-chartjs-2`（Chart.js 的 React 封装）做时序数据可视化。历史图表组件支持 1 小时、3 小时、24 小时时间窗口切换和日期选择。一个关键技术细节是数据降采样——对于 24 小时视图，我在时间段内取平均值来保持图表流畅，而不是绘制数千个原始数据点。"

---

## 面试加分建议

| 策略 | 说明 |
|------|------|
| **讲决策而非语法** | 老师不会问你 `useState` 怎么写，会问你**为什么这么设计** |
| **准备一个"最难的 bug"故事** | 比如：轮询导致的内存泄漏、传感器离线检测的时间阈值调优 |
| **强调工程公式嵌入** | IAPWS IF-97 湿度比计算嵌入 React 组件非常独特，说明你不只是前端开发，还理解领域知识 |
| **如果被问到 Vue** | 强调你理解两者的核心差异（JSX vs Template、Composition API vs Hooks），并能根据场景选择合适的工具 |
| **连接到 HAI 研究** | "These same patterns — real-time data polling, conditional UI rendering, threshold-based alerts — are directly applicable to building experiment UIs that react to user behavior in real time." |
