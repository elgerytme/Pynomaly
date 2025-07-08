export default {
  title: 'Foundation/Typography',
  parameters: {
    docs: {
      description: {
        component: 'Accessible typography system with semantic hierarchy and responsive scaling.'
      }
    }
  }
};

// Typography scale template
const TypographyTemplate = ({ elements, title, description }) => {
  return `
    <div class="p-6">
      <h3 class="text-2xl font-semibold mb-2">${title}</h3>
      <p class="text-gray-600 mb-6">${description}</p>
      <div class="space-y-6">
        ${elements.map(element => `
          <div class="border-b border-gray-200 pb-4">
            <div class="${element.class}" style="${element.style || ''}">${element.text}</div>
            <div class="mt-2 text-sm text-gray-500">
              <code class="bg-gray-100 px-2 py-1 rounded">${element.class}</code>
              <span class="ml-4">${element.specs}</span>
              ${element.accessibility ? `<span class="ml-4 text-green-600">♿ ${element.accessibility}</span>` : ''}
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
};

// Display Typography
export const DisplayText = TypographyTemplate.bind({});
DisplayText.args = {
  title: 'Display Typography',
  description: 'Large, impactful text for hero sections and major headings.',
  elements: [
    {
      text: 'Anomaly Detection Platform',
      class: 'text-6xl font-bold leading-none tracking-tight',
      specs: '60px / Bold / -0.025em',
      accessibility: 'Use sparingly, ensure adequate contrast'
    },
    {
      text: 'Advanced ML Analytics',
      class: 'text-5xl font-bold leading-none tracking-tight',
      specs: '48px / Bold / -0.025em',
      accessibility: 'Suitable for page titles'
    },
    {
      text: 'Real-time Monitoring',
      class: 'text-4xl font-bold leading-tight tracking-tight',
      specs: '36px / Bold / -0.025em',
      accessibility: 'Good for section headers'
    }
  ]
};

DisplayText.parameters = {
  docs: {
    description: {
      story: 'Display typography for hero sections and major page elements. Use sparingly and ensure proper contrast ratios.'
    }
  }
};

// Headline Typography
export const Headlines = TypographyTemplate.bind({});
Headlines.args = {
  title: 'Headlines',
  description: 'Semantic heading hierarchy for content structure and navigation.',
  elements: [
    {
      text: 'Large Headline (H1)',
      class: 'text-3xl font-semibold leading-tight',
      specs: '30px / Semibold / 1.25',
      accessibility: 'Only one H1 per page'
    },
    {
      text: 'Medium Headline (H2)',
      class: 'text-2xl font-semibold leading-tight',
      specs: '24px / Semibold / 1.25',
      accessibility: 'Use for major sections'
    },
    {
      text: 'Small Headline (H3)',
      class: 'text-xl font-semibold leading-tight',
      specs: '20px / Semibold / 1.25',
      accessibility: 'Use for subsections'
    }
  ]
};

Headlines.parameters = {
  docs: {
    description: {
      story: 'Semantic heading hierarchy following accessibility best practices. Maintain logical order (H1 → H2 → H3) for screen readers.'
    }
  }
};

// Title Typography
export const Titles = TypographyTemplate.bind({});
Titles.args = {
  title: 'Titles',
  description: 'Medium-weight text for component headers and important labels.',
  elements: [
    {
      text: 'Large Title',
      class: 'text-lg font-medium leading-normal',
      specs: '18px / Medium / 1.5',
      accessibility: 'Good for card headers'
    },
    {
      text: 'Medium Title',
      class: 'text-base font-medium leading-normal',
      specs: '16px / Medium / 1.5',
      accessibility: 'Standard component titles'
    },
    {
      text: 'Small Title',
      class: 'text-sm font-medium leading-normal',
      specs: '14px / Medium / 1.5',
      accessibility: 'Compact interface elements'
    }
  ]
};

Titles.parameters = {
  docs: {
    description: {
      story: 'Title typography for component headers, form labels, and UI element titles. Medium weight provides good hierarchy without being too heavy.'
    }
  }
};

// Body Typography
export const BodyText = TypographyTemplate.bind({});
BodyText.args = {
  title: 'Body Text',
  description: 'Primary text styles for content, descriptions, and user interface text.',
  elements: [
    {
      text: 'Large body text is ideal for important content that needs to be easily readable. This size works well for introductory paragraphs, important descriptions, and primary content areas where readability is paramount.',
      class: 'text-base font-normal leading-relaxed',
      specs: '16px / Normal / 1.625',
      accessibility: 'Meets AA standards at 4.5:1 contrast'
    },
    {
      text: 'Medium body text is the standard size for most interface content. It provides excellent readability while being space-efficient for forms, descriptions, and general content areas.',
      class: 'text-sm font-normal leading-relaxed',
      specs: '14px / Normal / 1.625',
      accessibility: 'Good for most UI text content'
    },
    {
      text: 'Small body text should be used sparingly for secondary information, captions, and metadata. Ensure adequate contrast when using this size.',
      class: 'text-xs font-normal leading-relaxed',
      specs: '12px / Normal / 1.625',
      accessibility: 'Requires higher contrast ratios'
    }
  ]
};

BodyText.parameters = {
  docs: {
    description: {
      story: 'Body text styles for content and interface text. The relaxed line height (1.625) improves readability and accessibility.'
    }
  }
};

// Label Typography
export const Labels = TypographyTemplate.bind({});
Labels.args = {
  title: 'Labels',
  description: 'Compact, medium-weight text for form labels, buttons, and UI controls.',
  elements: [
    {
      text: 'Large Label',
      class: 'text-sm font-medium leading-none',
      specs: '14px / Medium / 1.0',
      accessibility: 'Good for form labels'
    },
    {
      text: 'Medium Label',
      class: 'text-xs font-medium leading-none',
      specs: '12px / Medium / 1.0',
      accessibility: 'Standard button and control text'
    },
    {
      text: 'SMALL LABEL',
      class: 'text-xs font-medium leading-none uppercase tracking-wide',
      specs: '12px / Medium / 1.0 / 0.05em',
      accessibility: 'Use for categories and tags'
    }
  ]
};

Labels.parameters = {
  docs: {
    description: {
      story: 'Label typography for form controls, buttons, and compact interface elements. Tight line height saves space while maintaining readability.'
    }
  }
};

// Code Typography
export const CodeText = () => {
  return `
    <div class="p-6">
      <h3 class="text-2xl font-semibold mb-2">Code Typography</h3>
      <p class="text-gray-600 mb-6">Monospace typography for code, data, and technical content.</p>

      <div class="space-y-6">
        <div class="border-b border-gray-200 pb-4">
          <code class="text-sm font-mono bg-gray-100 px-2 py-1 rounded">inline code example</code>
          <div class="mt-2 text-sm text-gray-500">
            <code class="bg-gray-100 px-2 py-1 rounded">font-mono text-sm</code>
            <span class="ml-4">14px / JetBrains Mono</span>
          </div>
        </div>

        <div class="border-b border-gray-200 pb-4">
          <pre class="bg-gray-900 text-green-400 p-4 rounded-lg text-sm font-mono overflow-x-auto"><code>// Code block example
function detectAnomalies(data) {
  return model.predict(data);
}</code></pre>
          <div class="mt-2 text-sm text-gray-500">
            <code class="bg-gray-100 px-2 py-1 rounded">font-mono text-sm</code>
            <span class="ml-4">14px / JetBrains Mono / Code blocks</span>
          </div>
        </div>

        <div class="border-b border-gray-200 pb-4">
          <div class="font-mono text-xs text-gray-600">2024-06-26 10:30:15 UTC</div>
          <div class="mt-2 text-sm text-gray-500">
            <code class="bg-gray-100 px-2 py-1 rounded">font-mono text-xs</code>
            <span class="ml-4">12px / JetBrains Mono / Timestamps</span>
          </div>
        </div>
      </div>
    </div>
  `;
};

CodeText.parameters = {
  docs: {
    description: {
      story: 'Monospace typography for code, timestamps, and technical data. Uses JetBrains Mono for optimal readability and character distinction.'
    }
  }
};

// Typography Accessibility Guide
export const AccessibilityGuide = () => {
  return `
    <div class="p-6 max-w-4xl">
      <h3 class="text-2xl font-semibold mb-6">Typography Accessibility Guidelines</h3>

      <div class="space-y-8">
        <section>
          <h4 class="text-lg font-medium mb-4">Readability Requirements</h4>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-green-50 border border-green-200 p-4 rounded-lg">
              <h5 class="font-medium mb-2 text-green-900">✅ Best Practices</h5>
              <ul class="text-sm space-y-1 text-green-800">
                <li>• Minimum 16px for body text</li>
                <li>• Line height 1.5 or greater</li>
                <li>• Adequate color contrast (4.5:1)</li>
                <li>• Logical heading hierarchy</li>
                <li>• Maximum 80 characters per line</li>
              </ul>
            </div>
            <div class="bg-blue-50 border border-blue-200 p-4 rounded-lg">
              <h5 class="font-medium mb-2 text-blue-900">📏 Size Guidelines</h5>
              <ul class="text-sm space-y-1 text-blue-800">
                <li>• Large text: 18px+ (3:1 contrast)</li>
                <li>• Normal text: 16px+ (4.5:1 contrast)</li>
                <li>• Small text: 14px+ (higher contrast)</li>
                <li>• Minimum: 12px (use sparingly)</li>
              </ul>
            </div>
          </div>
        </section>

        <section>
          <h4 class="text-lg font-medium mb-4">Semantic Heading Structure</h4>
          <div class="bg-gray-50 p-4 rounded-lg">
            <div class="space-y-2 text-sm">
              <div class="flex items-center">
                <span class="font-mono bg-gray-200 px-2 py-1 rounded mr-3">H1</span>
                <span>Page title (only one per page)</span>
              </div>
              <div class="flex items-center ml-4">
                <span class="font-mono bg-gray-200 px-2 py-1 rounded mr-3">H2</span>
                <span>Major sections</span>
              </div>
              <div class="flex items-center ml-8">
                <span class="font-mono bg-gray-200 px-2 py-1 rounded mr-3">H3</span>
                <span>Subsections</span>
              </div>
              <div class="flex items-center ml-12">
                <span class="font-mono bg-gray-200 px-2 py-1 rounded mr-3">H4</span>
                <span>Sub-subsections</span>
              </div>
            </div>
          </div>
        </section>

        <section>
          <h4 class="text-lg font-medium mb-4">Responsive Typography</h4>
          <div class="overflow-x-auto">
            <table class="w-full text-sm border border-gray-200 rounded-lg">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-4 py-2 text-left border-b">Element</th>
                  <th class="px-4 py-2 text-left border-b">Mobile</th>
                  <th class="px-4 py-2 text-left border-b">Tablet</th>
                  <th class="px-4 py-2 text-left border-b">Desktop</th>
                </tr>
              </thead>
              <tbody>
                <tr class="border-b">
                  <td class="px-4 py-2 font-medium">Display Large</td>
                  <td class="px-4 py-2">36px</td>
                  <td class="px-4 py-2">48px</td>
                  <td class="px-4 py-2">60px</td>
                </tr>
                <tr class="border-b">
                  <td class="px-4 py-2 font-medium">H1</td>
                  <td class="px-4 py-2">24px</td>
                  <td class="px-4 py-2">28px</td>
                  <td class="px-4 py-2">30px</td>
                </tr>
                <tr class="border-b">
                  <td class="px-4 py-2 font-medium">Body</td>
                  <td class="px-4 py-2">16px</td>
                  <td class="px-4 py-2">16px</td>
                  <td class="px-4 py-2">16px</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h4 class="text-lg font-medium mb-4">Font Loading Best Practices</h4>
          <div class="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
            <h5 class="font-medium mb-2 text-yellow-900">⚡ Performance</h5>
            <ul class="text-sm space-y-1 text-yellow-800">
              <li>• Use font-display: swap for custom fonts</li>
              <li>• Preload critical font files</li>
              <li>• Provide system font fallbacks</li>
              <li>• Optimize font file sizes (woff2)</li>
              <li>• Minimize layout shift during loading</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  `;
};

AccessibilityGuide.parameters = {
  docs: {
    description: {
      story: 'Comprehensive typography accessibility guidelines covering readability, semantic structure, and performance considerations.'
    }
  }
};
