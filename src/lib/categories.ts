// 文章分类：分类即标签（frontmatter 的 tags）。侧栏按声明顺序展示已知标签，
// 未知标签按字典序补在后面；无标签的文章归入未分类。
// 文件夹只做内部组织（src/content/blog/<folder>/），不进 URL 也不进分类。
import type { CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;

// 侧栏顺序即此处的声明顺序；新增文章标签若不在列表里会自动追加到后面
export const CATEGORY_ORDER: string[] = [
  '强化学习',
  '后训练',
  '大模型',
  '训练工程',
  '模型结构',
];

export const MISC_KEY = '未分类';

// URL slug：链接只用文件名（/blog/<slug>/），文件夹不进入 URL
export function slugOf(post: Post): string {
  return post.id.split('/').pop()!;
}

// 按标签分组：一篇文章可以属于多个分类（多标签即多分类）
export function groupByTag(posts: Post[]): Map<string, Post[]> {
  const groups = new Map<string, Post[]>();
  for (const p of posts) {
    const tags = p.data.tags.length > 0 ? p.data.tags : [MISC_KEY];
    for (const t of tags) {
      if (!groups.has(t)) groups.set(t, []);
      groups.get(t)!.push(p);
    }
  }
  const known = CATEGORY_ORDER.filter((t) => groups.has(t));
  const unknown = [...groups.keys()]
    .filter((t) => !CATEGORY_ORDER.includes(t) && t !== MISC_KEY)
    .sort();
  const ordered = groups.has(MISC_KEY) ? [...known, ...unknown, MISC_KEY] : [...known, ...unknown];
  return new Map(ordered.map((k) => [k, groups.get(k)!]));
}
