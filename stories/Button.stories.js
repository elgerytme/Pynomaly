/**
 * Button Component Stories
 * Interactive documentation for Pynomaly button components
 */

export default {
  title: 'Components/Button',
  component: 'button',
  tags: ['autodocs'],
  parameters: {
    docs: {
      description: {
        component: 'Button components for user interactions in the Pynomaly platform. Supports multiple variants, sizes, and states with full accessibility compliance.',
      },
    },
    a11y: {
      config: {
        rules: [
          {
            id: 'color-contrast',
            enabled: true,
          },
          {
            id: 'focus-visible',
            enabled: true,
          },
          {
            id: 'keyboard',
            enabled: true,
          },
        ],
      },
    },
  },
  argTypes: {
    variant: {
      control: { type: 'select' },
      options: ['primary', 'secondary', 'success', 'warning', 'danger', 'ghost', 'link'],
      description: 'Visual style variant',
    },
    size: {
      control: { type: 'select' },
      options: ['xs', 'sm', 'base', 'lg', 'xl'],
      description: 'Button size',
    },
    disabled: {
      control: { type: 'boolean' },
      description: 'Disable button interaction',
    },
    loading: {
      control: { type: 'boolean' },
      description: 'Show loading state',
    },
    fullWidth: {
      control: { type: 'boolean' },
      description: 'Make button full width',
    },
    icon: {
      control: { type: 'text' },
      description: 'Icon name (optional)',
    },
    iconPosition: {
      control: { type: 'select' },
      options: ['left', 'right'],
      description: 'Icon position relative to text',
    },
    children: {
      control: { type: 'text' },
      description: 'Button text content',
    },
  },
};

// Helper function to create button HTML
const createButton = ({
  variant = 'primary',
  size = 'base',
  disabled = false,
  loading = false,
  fullWidth = false,
  icon = '',
  iconPosition = 'left',
  children = 'Button',
  onClick = () => {},
  ...props
}) => {
  const button = document.createElement('button');
  
  // Base classes
  const baseClasses = [
    'btn',
    'inline-flex',
    'items-center',
    'justify-center',
    'rounded-lg',
    'font-medium',
    'transition-all',
    'duration-200',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-offset-2',
    'disabled:opacity-50',
    'disabled:cursor-not-allowed',
  ];
  
  // Variant classes
  const variantClasses = {
    primary: [
      'bg-blue-600', 'text-white', 'border-blue-600',
      'hover:bg-blue-700', 'focus:ring-blue-500',
      'active:bg-blue-800'
    ],
    secondary: [
      'bg-gray-100', 'text-gray-900', 'border-gray-300',
      'hover:bg-gray-200', 'focus:ring-gray-500',
      'active:bg-gray-300'
    ],
    success: [
      'bg-green-600', 'text-white', 'border-green-600',
      'hover:bg-green-700', 'focus:ring-green-500',
      'active:bg-green-800'
    ],
    warning: [
      'bg-yellow-500', 'text-white', 'border-yellow-500',
      'hover:bg-yellow-600', 'focus:ring-yellow-400',
      'active:bg-yellow-700'
    ],
    danger: [
      'bg-red-600', 'text-white', 'border-red-600',
      'hover:bg-red-700', 'focus:ring-red-500',
      'active:bg-red-800'
    ],
    ghost: [
      'bg-transparent', 'text-gray-700', 'border-transparent',
      'hover:bg-gray-100', 'focus:ring-gray-500',
      'active:bg-gray-200'
    ],
    link: [
      'bg-transparent', 'text-blue-600', 'border-transparent',
      'hover:text-blue-700', 'focus:ring-blue-500',
      'underline', 'active:text-blue-800'
    ],
  };
  
  // Size classes
  const sizeClasses = {
    xs: ['px-2', 'py-1', 'text-xs', 'min-h-[24px]'],
    sm: ['px-3', 'py-1.5', 'text-sm', 'min-h-[32px]'],
    base: ['px-4', 'py-2', 'text-base', 'min-h-[40px]'],
    lg: ['px-6', 'py-3', 'text-lg', 'min-h-[48px]'],
    xl: ['px-8', 'py-4', 'text-xl', 'min-h-[56px]'],
  };
  
  // Apply classes
  const allClasses = [
    ...baseClasses,
    ...variantClasses[variant],
    ...sizeClasses[size],
    fullWidth ? 'w-full' : '',
    loading ? 'cursor-wait' : '',
  ].filter(Boolean);
  
  button.className = allClasses.join(' ');
  
  // Set attributes
  button.type = 'button';
  button.disabled = disabled || loading;
  
  // Accessibility attributes
  if (loading) {
    button.setAttribute('aria-busy', 'true');
    button.setAttribute('aria-describedby', 'loading-description');
  }
  
  if (disabled) {
    button.setAttribute('aria-disabled', 'true');
  }
  
  // Create content
  const content = document.createElement('span');
  content.className = 'flex items-center justify-center gap-2';
  
  // Add loading spinner
  if (loading) {
    const spinner = document.createElement('svg');
    spinner.className = 'animate-spin h-4 w-4';
    spinner.innerHTML = `
      <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" opacity="0.25"></circle>
      <path fill="currentColor" opacity="0.75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    `;
    spinner.setAttribute('viewBox', '0 0 24 24');
    content.appendChild(spinner);
  }
  
  // Add icon (if provided and not loading)
  if (icon && !loading) {
    const iconElement = document.createElement('span');
    iconElement.className = 'icon';
    iconElement.setAttribute('data-icon', icon);
    iconElement.innerHTML = getIconSvg(icon);
    
    if (iconPosition === 'left') {
      content.appendChild(iconElement);
    }
  }
  
  // Add text
  if (children) {
    const textElement = document.createElement('span');
    textElement.textContent = loading ? 'Loading...' : children;
    content.appendChild(textElement);
  }
  
  // Add icon (right position)
  if (icon && !loading && iconPosition === 'right') {
    const iconElement = document.createElement('span');
    iconElement.className = 'icon';
    iconElement.setAttribute('data-icon', icon);
    iconElement.innerHTML = getIconSvg(icon);
    content.appendChild(iconElement);
  }
  
  button.appendChild(content);
  
  // Add event listeners
  button.addEventListener('click', onClick);
  
  // Add accessibility enhancements
  button.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!disabled && !loading) {
        onClick(e);
      }
    }
  });
  
  return button;
};

// Helper function to get icon SVG
const getIconSvg = (iconName) => {
  const icons = {
    plus: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>`,
    check: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`,
    x: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>`,
    download: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2z"></path></svg>`,
    upload: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>`,
  };
  return icons[iconName] || '';
};

// Story templates
export const Primary = {
  args: {
    children: 'Primary Button',
    variant: 'primary',
    size: 'base',
  },
  render: createButton,
};

export const Secondary = {
  args: {
    children: 'Secondary Button',
    variant: 'secondary',
    size: 'base',
  },
  render: createButton,
};

export const Success = {
  args: {
    children: 'Success Button',
    variant: 'success',
    size: 'base',
    icon: 'check',
  },
  render: createButton,
};

export const Warning = {
  args: {
    children: 'Warning Button',
    variant: 'warning',
    size: 'base',
  },
  render: createButton,
};

export const Danger = {
  args: {
    children: 'Danger Button',
    variant: 'danger',
    size: 'base',
    icon: 'x',
  },
  render: createButton,
};

export const Ghost = {
  args: {
    children: 'Ghost Button',
    variant: 'ghost',
    size: 'base',
  },
  render: createButton,
};

export const Link = {
  args: {
    children: 'Link Button',
    variant: 'link',
    size: 'base',
  },
  render: createButton,
};

export const Loading = {
  args: {
    children: 'Loading Button',
    variant: 'primary',
    size: 'base',
    loading: true,
  },
  render: createButton,
};

export const Disabled = {
  args: {
    children: 'Disabled Button',
    variant: 'primary',
    size: 'base',
    disabled: true,
  },
  render: createButton,
};

export const WithIcon = {
  args: {
    children: 'Download',
    variant: 'primary',
    size: 'base',
    icon: 'download',
    iconPosition: 'left',
  },
  render: createButton,
};

export const IconRight = {
  args: {
    children: 'Upload',
    variant: 'secondary',
    size: 'base',
    icon: 'upload',
    iconPosition: 'right',
  },
  render: createButton,
};

export const FullWidth = {
  args: {
    children: 'Full Width Button',
    variant: 'primary',
    size: 'base',
    fullWidth: true,
  },
  render: createButton,
};

export const Sizes = {
  render: () => {
    const container = document.createElement('div');
    container.className = 'space-y-4';
    
    const sizes = ['xs', 'sm', 'base', 'lg', 'xl'];
    sizes.forEach(size => {
      const button = createButton({
        children: `${size.toUpperCase()} Button`,
        variant: 'primary',
        size,
      });
      container.appendChild(button);
    });
    
    return container;
  },
  parameters: {
    docs: {
      description: {
        story: 'Different button sizes available in the design system.',
      },
    },
  },
};

export const Variants = {
  render: () => {
    const container = document.createElement('div');
    container.className = 'flex flex-wrap gap-4';
    
    const variants = ['primary', 'secondary', 'success', 'warning', 'danger', 'ghost', 'link'];
    variants.forEach(variant => {
      const button = createButton({
        children: variant.charAt(0).toUpperCase() + variant.slice(1),
        variant,
        size: 'base',
      });
      container.appendChild(button);
    });
    
    return container;
  },
  parameters: {
    docs: {
      description: {
        story: 'All available button variants with their respective styling.',
      },
    },
  },
};

export const InteractiveExample = {
  args: {
    children: 'Click me!',
    variant: 'primary',
    size: 'base',
    onClick: () => alert('Button clicked!'),
  },
  render: createButton,
  parameters: {
    docs: {
      description: {
        story: 'Interactive button example with click handler. Try clicking the button!',
      },
    },
  },
};