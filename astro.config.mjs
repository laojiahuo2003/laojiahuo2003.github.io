import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://laojiahuo2003.github.io',
  integrations: [sitemap()],
});
