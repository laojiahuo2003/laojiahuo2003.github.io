import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).default([]),
  }),
});

// 项目经历：简历素材库。正文用 markdown 列表写简历要点，
// 页面渲染成可一键复制纯文本的条目。
const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    role: z.string().default(''),
    period: z.string().default(''),        // 如 2026.03 - 至今
    stack: z.array(z.string()).default([]),
    links: z.array(z.object({ label: z.string(), url: z.string() })).default([]),
    order: z.number().default(0),          // 越大越靠前
  }),
});

export const collections = { blog, projects };
