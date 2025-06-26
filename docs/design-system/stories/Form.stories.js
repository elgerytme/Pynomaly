/**
 * Form Components Stories
 * Interactive documentation for Pynomaly form components with accessibility focus
 */

export default {
  title: 'Components/Form',
  tags: ['autodocs'],
  parameters: {
    docs: {
      description: {
        component: 'Accessible form components for data input in the Pynomaly platform. All components follow WCAG 2.1 AA guidelines with proper labeling, validation, and error handling.',
      },
    },
    a11y: {
      config: {
        rules: [
          { id: 'label', enabled: true },
          { id: 'form-field-multiple-labels', enabled: true },
          { id: 'landmark-one-main', enabled: false },
        ],
      },
    },
  },
};

// Form Input Component
const createInput = ({
  id = 'input-' + Math.random().toString(36).substr(2, 9),
  label = 'Input Label',
  type = 'text',
  placeholder = '',
  value = '',
  disabled = false,
  required = false,
  error = '',
  helpText = '',
  size = 'base',
  fullWidth = true,
  ...props
}) => {
  const container = document.createElement('div');
  container.className = `form-field ${fullWidth ? 'w-full' : ''}`;
  
  // Create label
  const labelElement = document.createElement('label');
  labelElement.htmlFor = id;
  labelElement.className = 'block text-sm font-medium text-gray-700 mb-1';
  labelElement.textContent = label;
  if (required) {
    const requiredSpan = document.createElement('span');
    requiredSpan.className = 'text-red-500 ml-1';
    requiredSpan.textContent = '*';
    requiredSpan.setAttribute('aria-label', 'required');
    labelElement.appendChild(requiredSpan);
  }
  
  // Create input
  const input = document.createElement('input');
  input.type = type;
  input.id = id;
  input.name = id;
  input.value = value;
  input.placeholder = placeholder;
  input.disabled = disabled;
  input.required = required;
  
  // Size classes
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    base: 'px-3 py-2 text-base',
    lg: 'px-4 py-3 text-lg',
  };
  
  // Base input classes
  const inputClasses = [
    'block',
    'w-full',
    'border',
    'rounded-md',
    'shadow-sm',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-blue-500',
    'disabled:bg-gray-50',
    'disabled:text-gray-500',
    'disabled:cursor-not-allowed',
    'transition-colors',
    'duration-200',
    sizeClasses[size],
  ];
  
  // Error state classes
  if (error) {
    inputClasses.push(
      'border-red-300',
      'text-red-900',
      'placeholder-red-300',
      'focus:ring-red-500',
      'focus:border-red-500'
    );
  } else {
    inputClasses.push('border-gray-300');
  }
  
  input.className = inputClasses.join(' ');
  
  // Accessibility attributes
  const describedBy = [];
  if (helpText) describedBy.push(`${id}-help`);
  if (error) describedBy.push(`${id}-error`);
  if (describedBy.length > 0) {
    input.setAttribute('aria-describedby', describedBy.join(' '));
  }
  if (error) {
    input.setAttribute('aria-invalid', 'true');
  }
  
  // Assemble the component
  container.appendChild(labelElement);
  container.appendChild(input);
  
  // Add help text
  if (helpText) {
    const helpElement = document.createElement('p');
    helpElement.id = `${id}-help`;
    helpElement.className = 'mt-1 text-sm text-gray-500';
    helpElement.textContent = helpText;
    container.appendChild(helpElement);
  }
  
  // Add error message
  if (error) {
    const errorElement = document.createElement('p');
    errorElement.id = `${id}-error`;
    errorElement.className = 'mt-1 text-sm text-red-600';
    errorElement.setAttribute('role', 'alert');
    errorElement.textContent = error;
    container.appendChild(errorElement);
  }
  
  return container;
};

// Select Component
const createSelect = ({
  id = 'select-' + Math.random().toString(36).substr(2, 9),
  label = 'Select Label',
  options = [],
  value = '',
  disabled = false,
  required = false,
  error = '',
  helpText = '',
  placeholder = 'Select an option...',
  fullWidth = true,
  ...props
}) => {
  const container = document.createElement('div');
  container.className = `form-field ${fullWidth ? 'w-full' : ''}`;
  
  // Create label
  const labelElement = document.createElement('label');
  labelElement.htmlFor = id;
  labelElement.className = 'block text-sm font-medium text-gray-700 mb-1';
  labelElement.textContent = label;
  if (required) {
    const requiredSpan = document.createElement('span');
    requiredSpan.className = 'text-red-500 ml-1';
    requiredSpan.textContent = '*';
    requiredSpan.setAttribute('aria-label', 'required');
    labelElement.appendChild(requiredSpan);
  }
  
  // Create select
  const select = document.createElement('select');
  select.id = id;
  select.name = id;
  select.value = value;
  select.disabled = disabled;
  select.required = required;
  
  // Base select classes
  const selectClasses = [
    'block',
    'w-full',
    'px-3',
    'py-2',
    'border',
    'rounded-md',
    'shadow-sm',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-blue-500',
    'disabled:bg-gray-50',
    'disabled:text-gray-500',
    'disabled:cursor-not-allowed',
    'bg-white',
    'text-base',
  ];
  
  // Error state classes
  if (error) {
    selectClasses.push(
      'border-red-300',
      'text-red-900',
      'focus:ring-red-500',
      'focus:border-red-500'
    );
  } else {
    selectClasses.push('border-gray-300');
  }
  
  select.className = selectClasses.join(' ');
  
  // Add placeholder option
  if (placeholder) {
    const placeholderOption = document.createElement('option');
    placeholderOption.value = '';
    placeholderOption.textContent = placeholder;
    placeholderOption.disabled = true;
    placeholderOption.selected = !value;
    select.appendChild(placeholderOption);
  }
  
  // Add options
  options.forEach(option => {
    const optionElement = document.createElement('option');
    optionElement.value = option.value || option;
    optionElement.textContent = option.label || option;
    optionElement.selected = optionElement.value === value;
    select.appendChild(optionElement);
  });
  
  // Accessibility attributes
  const describedBy = [];
  if (helpText) describedBy.push(`${id}-help`);
  if (error) describedBy.push(`${id}-error`);
  if (describedBy.length > 0) {
    select.setAttribute('aria-describedby', describedBy.join(' '));
  }
  if (error) {
    select.setAttribute('aria-invalid', 'true');
  }
  
  // Assemble the component
  container.appendChild(labelElement);
  container.appendChild(select);
  
  // Add help text
  if (helpText) {
    const helpElement = document.createElement('p');
    helpElement.id = `${id}-help`;
    helpElement.className = 'mt-1 text-sm text-gray-500';
    helpElement.textContent = helpText;
    container.appendChild(helpElement);
  }
  
  // Add error message
  if (error) {
    const errorElement = document.createElement('p');
    errorElement.id = `${id}-error`;
    errorElement.className = 'mt-1 text-sm text-red-600';
    errorElement.setAttribute('role', 'alert');
    errorElement.textContent = error;
    container.appendChild(errorElement);
  }
  
  return container;
};

// Textarea Component
const createTextarea = ({
  id = 'textarea-' + Math.random().toString(36).substr(2, 9),
  label = 'Textarea Label',
  placeholder = '',
  value = '',
  disabled = false,
  required = false,
  error = '',
  helpText = '',
  rows = 4,
  fullWidth = true,
  ...props
}) => {
  const container = document.createElement('div');
  container.className = `form-field ${fullWidth ? 'w-full' : ''}`;
  
  // Create label
  const labelElement = document.createElement('label');
  labelElement.htmlFor = id;
  labelElement.className = 'block text-sm font-medium text-gray-700 mb-1';
  labelElement.textContent = label;
  if (required) {
    const requiredSpan = document.createElement('span');
    requiredSpan.className = 'text-red-500 ml-1';
    requiredSpan.textContent = '*';
    requiredSpan.setAttribute('aria-label', 'required');
    labelElement.appendChild(requiredSpan);
  }
  
  // Create textarea
  const textarea = document.createElement('textarea');
  textarea.id = id;
  textarea.name = id;
  textarea.value = value;
  textarea.placeholder = placeholder;
  textarea.disabled = disabled;
  textarea.required = required;
  textarea.rows = rows;
  
  // Base textarea classes
  const textareaClasses = [
    'block',
    'w-full',
    'px-3',
    'py-2',
    'border',
    'rounded-md',
    'shadow-sm',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-blue-500',
    'disabled:bg-gray-50',
    'disabled:text-gray-500',
    'disabled:cursor-not-allowed',
    'resize-vertical',
    'text-base',
  ];
  
  // Error state classes
  if (error) {
    textareaClasses.push(
      'border-red-300',
      'text-red-900',
      'placeholder-red-300',
      'focus:ring-red-500',
      'focus:border-red-500'
    );
  } else {
    textareaClasses.push('border-gray-300');
  }
  
  textarea.className = textareaClasses.join(' ');
  
  // Accessibility attributes
  const describedBy = [];
  if (helpText) describedBy.push(`${id}-help`);
  if (error) describedBy.push(`${id}-error`);
  if (describedBy.length > 0) {
    textarea.setAttribute('aria-describedby', describedBy.join(' '));
  }
  if (error) {
    textarea.setAttribute('aria-invalid', 'true');
  }
  
  // Assemble the component
  container.appendChild(labelElement);
  container.appendChild(textarea);
  
  // Add help text
  if (helpText) {
    const helpElement = document.createElement('p');
    helpElement.id = `${id}-help`;
    helpElement.className = 'mt-1 text-sm text-gray-500';
    helpElement.textContent = helpText;
    container.appendChild(helpElement);
  }
  
  // Add error message
  if (error) {
    const errorElement = document.createElement('p');
    errorElement.id = `${id}-error`;
    errorElement.className = 'mt-1 text-sm text-red-600';
    errorElement.setAttribute('role', 'alert');
    errorElement.textContent = error;
    container.appendChild(errorElement);
  }
  
  return container;
};

// Complete Form Example
const createCompleteForm = () => {
  const form = document.createElement('form');
  form.className = 'max-w-md mx-auto space-y-6 p-6 bg-white rounded-lg shadow-lg';
  form.setAttribute('novalidate', '');
  
  // Form title
  const title = document.createElement('h2');
  title.className = 'text-2xl font-bold text-gray-900 mb-6';
  title.textContent = 'Dataset Configuration';
  form.appendChild(title);
  
  // Dataset name input
  const nameField = createInput({
    id: 'dataset-name',
    label: 'Dataset Name',
    placeholder: 'Enter dataset name',
    required: true,
    helpText: 'A descriptive name for your dataset',
  });
  form.appendChild(nameField);
  
  // Algorithm selection
  const algorithmField = createSelect({
    id: 'algorithm',
    label: 'Detection Algorithm',
    required: true,
    options: [
      { value: 'isolation-forest', label: 'Isolation Forest' },
      { value: 'one-class-svm', label: 'One-Class SVM' },
      { value: 'local-outlier-factor', label: 'Local Outlier Factor' },
      { value: 'ensemble', label: 'Ensemble Methods' },
    ],
    helpText: 'Choose the anomaly detection algorithm',
  });
  form.appendChild(algorithmField);
  
  // Contamination rate
  const contaminationField = createInput({
    id: 'contamination',
    label: 'Contamination Rate',
    type: 'number',
    placeholder: '0.1',
    helpText: 'Expected proportion of anomalies (0.0 to 0.5)',
  });
  form.appendChild(contaminationField);
  
  // Description
  const descriptionField = createTextarea({
    id: 'description',
    label: 'Description',
    placeholder: 'Describe your dataset and analysis goals...',
    rows: 3,
    helpText: 'Optional description of the dataset and analysis objectives',
  });
  form.appendChild(descriptionField);
  
  // Buttons
  const buttonGroup = document.createElement('div');
  buttonGroup.className = 'flex gap-3 pt-4';
  
  const submitButton = document.createElement('button');
  submitButton.type = 'submit';
  submitButton.className = 'flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors';
  submitButton.textContent = 'Create Dataset';
  
  const cancelButton = document.createElement('button');
  cancelButton.type = 'button';
  cancelButton.className = 'px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors';
  cancelButton.textContent = 'Cancel';
  
  buttonGroup.appendChild(submitButton);
  buttonGroup.appendChild(cancelButton);
  form.appendChild(buttonGroup);
  
  // Form validation
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Simple validation example
    const nameInput = form.querySelector('#dataset-name');
    const algorithmSelect = form.querySelector('#algorithm');
    
    let hasErrors = false;
    
    // Remove existing errors
    form.querySelectorAll('[role="alert"]').forEach(el => el.remove());
    form.querySelectorAll('[aria-invalid="true"]').forEach(el => {
      el.setAttribute('aria-invalid', 'false');
      el.className = el.className.replace(/border-red-\d+|text-red-\d+|focus:ring-red-\d+|focus:border-red-\d+/g, '');
      el.className += ' border-gray-300';
    });
    
    // Validate name
    if (!nameInput.value.trim()) {
      hasErrors = true;
      nameInput.setAttribute('aria-invalid', 'true');
      nameInput.className = nameInput.className.replace('border-gray-300', 'border-red-300 text-red-900 focus:ring-red-500 focus:border-red-500');
      
      const errorElement = document.createElement('p');
      errorElement.className = 'mt-1 text-sm text-red-600';
      errorElement.setAttribute('role', 'alert');
      errorElement.textContent = 'Dataset name is required';
      nameInput.parentNode.appendChild(errorElement);
    }
    
    // Validate algorithm
    if (!algorithmSelect.value) {
      hasErrors = true;
      algorithmSelect.setAttribute('aria-invalid', 'true');
      algorithmSelect.className = algorithmSelect.className.replace('border-gray-300', 'border-red-300 text-red-900 focus:ring-red-500 focus:border-red-500');
      
      const errorElement = document.createElement('p');
      errorElement.className = 'mt-1 text-sm text-red-600';
      errorElement.setAttribute('role', 'alert');
      errorElement.textContent = 'Please select an algorithm';
      algorithmSelect.parentNode.appendChild(errorElement);
    }
    
    if (!hasErrors) {
      alert('Form submitted successfully!');
    }
  });
  
  return form;
};

// Stories
export const TextInput = {
  render: () => createInput({
    label: 'Dataset Name',
    placeholder: 'Enter dataset name',
    helpText: 'A descriptive name for your dataset',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Basic text input with label and help text.',
      },
    },
  },
};

export const RequiredInput = {
  render: () => createInput({
    label: 'Required Field',
    placeholder: 'This field is required',
    required: true,
    helpText: 'This field must be filled out',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Required input field with visual indicator.',
      },
    },
  },
};

export const InputWithError = {
  render: () => createInput({
    label: 'Dataset Name',
    placeholder: 'Enter dataset name',
    value: '',
    required: true,
    error: 'Dataset name is required',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Input field showing validation error state.',
      },
    },
  },
};

export const DisabledInput = {
  render: () => createInput({
    label: 'Disabled Field',
    value: 'This field is disabled',
    disabled: true,
    helpText: 'This field cannot be edited',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Disabled input field.',
      },
    },
  },
};

export const SelectDropdown = {
  render: () => createSelect({
    label: 'Detection Algorithm',
    required: true,
    options: [
      { value: 'isolation-forest', label: 'Isolation Forest' },
      { value: 'one-class-svm', label: 'One-Class SVM' },
      { value: 'local-outlier-factor', label: 'Local Outlier Factor' },
      { value: 'ensemble', label: 'Ensemble Methods' },
    ],
    helpText: 'Choose the anomaly detection algorithm',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Select dropdown with multiple options.',
      },
    },
  },
};

export const TextareaField = {
  render: () => createTextarea({
    label: 'Description',
    placeholder: 'Describe your dataset and analysis goals...',
    rows: 4,
    helpText: 'Optional description of the dataset and analysis objectives',
  }),
  parameters: {
    docs: {
      description: {
        story: 'Textarea for longer text input.',
      },
    },
  },
};

export const FormInputSizes = {
  render: () => {
    const container = document.createElement('div');
    container.className = 'space-y-4';
    
    const sizes = ['sm', 'base', 'lg'];
    sizes.forEach(size => {
      const input = createInput({
        label: `${size.toUpperCase()} Size Input`,
        placeholder: `${size} sized input`,
        size,
      });
      container.appendChild(input);
    });
    
    return container;
  },
  parameters: {
    docs: {
      description: {
        story: 'Different input sizes available in the design system.',
      },
    },
  },
};

export const CompleteForm = {
  render: createCompleteForm,
  parameters: {
    docs: {
      description: {
        story: 'Complete form example with validation, showing how all form components work together. Try submitting without filling required fields to see validation in action.',
      },
    },
  },
};