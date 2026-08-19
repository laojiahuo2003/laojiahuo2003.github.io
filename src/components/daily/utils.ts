// 日报渲染工具：语言色、格式化、期号计算。
// 数据结构由 daily/ 生成器的 feed.json（src/data/feed.json）提供。

export interface Repo {
  name: string;
  url?: string;
  stars?: number;
  stars_fmt?: string;
  growth?: number;
  weekly_growth?: number;
  language?: string;
  category?: string;
  description?: string;
  created?: string;
  source?: string;
}

export interface Report {
  date: string;
  best?: Repo;
  leaderboard?: Repo[];
  fast_growing?: Repo[];
  monthly?: Repo[];
  by_category?: { category: string; projects: Repo[] }[];
  new_projects?: { category: string; projects: Repo[] }[];
  explored?: { strategy: string; projects: Repo[] }[];
  newly_discovered?: Repo[];
}

export interface Feed {
  updated: string;
  reports: Report[];
}

// 期号纪元：旧仓库（github-daily-report）首期日期，期号 = 距此天数 + 1，
// 保证迁移前后期号连续。
export const ISSUE_EPOCH = '2026-04-02';

export function issueNo(date: string): number {
  const days = Math.round((Date.parse(date) - Date.parse(ISSUE_EPOCH)) / 86400000);
  return days + 1;
}

// GitHub 官方语言色（常用子集，未命中则不着色）
export const LANG_COLORS: Record<string, string> = {
  TypeScript: '#3178c6', JavaScript: '#f1e05a', Python: '#3572A5',
  Rust: '#dea584', Go: '#00ADD8', 'C++': '#f34b7d', C: '#555555',
  'C#': '#178600', Java: '#b07219', Ruby: '#701516', PHP: '#4F5D95',
  Swift: '#F05138', Kotlin: '#A97BFF', Dart: '#00B4AB', Vue: '#41b883',
  HTML: '#e34c26', CSS: '#663399', SCSS: '#c6538c', Shell: '#89e051',
  Lua: '#000080', Haskell: '#5e5086', Elixir: '#6e4a7e', Zig: '#ec915c',
  R: '#198CE7', Scala: '#c22d40', Clojure: '#db5855', Perl: '#0298c3',
  'Jupyter Notebook': '#DA5B0B', Nix: '#7e7eff', OCaml: '#ef7a08',
  Solidity: '#AA6746', Assembly: '#6E4C13', PowerShell: '#012456',
  CoffeeScript: '#244776', MDX: '#fcb32c',
};

export function langColor(lang?: string): string {
  return lang ? LANG_COLORS[lang] ?? '' : '';
}

// 分类名自带 emoji 前缀（如 "🤖 AI / LLM"），期刊标题里只留文字
export function cleanCat(c?: string): string {
  return String(c ?? '')
    .replace(/^[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}\u{200D}]+\s*/u, '')
    .trim();
}

export function safeUrl(u?: string): string {
  return /^https?:\/\//.test(u ?? '') ? (u as string) : '';
}

export function fmtStars(n?: number): string {
  n = +n || 0;
  if (n >= 1000) {
    const s = (n / 1000).toFixed(1);
    return (s.endsWith('.0') ? s.slice(0, -2) : s) + 'k';
  }
  return String(n);
}

export function fmtDate(d?: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d ?? '');
  return m ? `${+m[1]} 年 ${+m[2]} 月 ${+m[3]} 日` : (d ?? '');
}

// 增长数值：优先日增长，回退周增长
export function growOf(r: Repo): number {
  return +r.growth > 0 ? +r.growth : +r.weekly_growth > 0 ? +r.weekly_growth : 0;
}

// 整期统计：项目总数与分区数（报头 statline 用）
export function reportStats(r: Report): { total: number; sections: number } {
  let total = 0;
  let sections = 0;
  const add = (projects?: Repo[]) => {
    if (projects?.length) { total += projects.length; sections++; }
  };
  if (r.best) { total += 1; sections++; }
  add(r.leaderboard);
  add(r.fast_growing);
  add(r.monthly);
  (r.by_category ?? []).forEach((g) => add(g.projects));
  (r.new_projects ?? []).forEach((g) => add(g.projects));
  (r.explored ?? []).forEach((g) => add(g.projects));
  add(r.newly_discovered);
  return { total, sections };
}
