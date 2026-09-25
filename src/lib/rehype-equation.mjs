// 块级公式的编号与溢出处理。
//
// 背景：KaTeX 把 \tag 渲染成 .katex-html 下的一个行内元素（.tag），并用 CSS 的
// position:absolute; right:0 把它钉到公式盒子的右缘。这带来两个问题：
//   1. 公式盒子的宽度就是正文栏宽，公式本身又是 white-space:nowrap，
//      一旦公式比正文栏宽，它会向两侧溢出，而编号仍固定在容器右缘 —— 两者必然重叠；
//   2. 编号在公式盒子内部，无法单独定位或留白。
//
// 处理方式：把编号从公式盒子里「搬出来」，变成 flex 布局里的兄弟节点：
//
//   <div class="equation">
//     <div class="equation-body">…katex-display…</div>   ← 过宽时只在这里横向滚动
//     <span class="equation-number">(4)</span>           ← 永不与公式重叠
//   </div>
//
// 同时把所有块级公式都套上这一层，保证行距一致。

// KaTeX 各版本对编号元素用的类名不同（0.16 用 .tag，0.18 用 .katex-tag），两个都认
const NUMBER_CLASSES = new Set(['tag', 'katex-tag']);

function classesOf(node) {
  const value = node.properties?.className;
  if (!value) return [];
  return Array.isArray(value) ? value.map(String) : String(value).split(/\s+/);
}

function textOf(node) {
  if (node.type === 'text') return node.value;
  if (!node.children) return '';
  return node.children.map(textOf).join('');
}

/** 深度优先，摘掉第一个编号元素并返回它的文字 */
function takeNumber(node) {
  if (!node.children) return '';
  let found = '';
  node.children = node.children.filter((child) => {
    if (found) return true;
    if (child.type === 'element' && classesOf(child).some((c) => NUMBER_CLASSES.has(c))) {
      found = textOf(child).trim();
      return false;
    }
    found = takeNumber(child);
    return true;
  });
  return found;
}

export default function rehypeEquation() {
  return (tree) => {
    const walk = (node) => {
      if (!node.children) return;
      node.children = node.children.flatMap((child) => {
        const isDisplayMath =
          child.type === 'element' && classesOf(child).includes('katex-display');

        if (!isDisplayMath) {
          walk(child);
          return [child];
        }

        const number = takeNumber(child);

        return [
          {
            type: 'element',
            tagName: 'div',
            properties: { className: ['equation'] },
            children: [
              {
                type: 'element',
                tagName: 'div',
                properties: { className: ['equation-body'] },
                children: [child],
              },
              ...(number
                ? [
                    {
                      type: 'element',
                      tagName: 'span',
                      properties: { className: ['equation-number'] },
                      children: [{ type: 'text', value: number }],
                    },
                  ]
                : []),
            ],
          },
        ];
      });
    };

    walk(tree);
  };
}
