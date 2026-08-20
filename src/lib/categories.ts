// 文章分类：文件夹即分类（src/content/blog/<folder>/*.md），根目录文件归入 misc。
// URL 只用文件名（/blog/<slug>/），文件夹不进 URL——加分类后外链不失效。
import type { CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;

// 展示名映射；侧栏顺序即此处的声明顺序
export const CATEGORIES: Record<string, string> = {
  inference: '推理',
  architecture: '模型结构',
  training: '训练',
  agent: 'Agent',
  basics: '基础',
  automation: '自动化',
};

export const MISC_KEY = 'misc';
export const MISC_LABEL = '未分类';

export function categoryOf(post: Post): string {
  return post.id.includes('/') ? post.id.split('/')[0] : MISC_KEY;
}

export function categoryLabel(key: string): string {
  if (key === MISC_KEY) return MISC_LABEL;
  return CATEGORIES[key] ?? key;
}

// URL slug：文件夹路径只做分类，链接只用文件名
export function slugOf(post: Post): string {
  return post.id.split('/').pop()!;
}

// 按声明顺序（未知文件夹按字典序补在后面，misc 永远最后）分组
export function groupByCategory(posts: Post[]): Map<string, Post[]> {
  const groups = new Map<string, Post[]>();
  for (const p of posts) {
    const k = categoryOf(p);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(p);
  }
  const known = Object.keys(CATEGORIES).filter((k) => groups.has(k));
  const unknown = [...groups.keys()].filter((k) => !(k in CATEGORIES) && k !== MISC_KEY).sort();
  const ordered = groups.has(MISC_KEY) ? [...known, ...unknown, MISC_KEY] : [...known, ...unknown];
  return new Map(ordered.map((k) => [k, groups.get(k)!]));
}
