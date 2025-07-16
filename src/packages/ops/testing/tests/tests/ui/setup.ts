import { vi, beforeEach, afterEach } from 'vitest'
import '@testing-library/jest-dom'

// Mock global objects
Object.defineProperty(window, 'ResizeObserver', {
  writable: true,
  value: vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  })),
})

Object.defineProperty(window, 'IntersectionObserver', {
  writable: true,
  value: vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  })),
})

// Mock Chart.js
vi.mock('chart.js', () => ({
  Chart: vi.fn().mockImplementation(() => ({
    destroy: vi.fn(),
    update: vi.fn(),
    render: vi.fn(),
    data: { datasets: [] },
    options: {}
  })),
  registerables: []
}))

// Mock D3
vi.mock('d3', () => ({
  select: vi.fn(() => ({
    selectAll: vi.fn(() => ({ remove: vi.fn() })),
    append: vi.fn(() => ({ attr: vi.fn(), style: vi.fn() })),
    attr: vi.fn(),
    style: vi.fn(),
    text: vi.fn(),
    data: vi.fn(() => ({ enter: vi.fn(), exit: vi.fn() }))
  })),
  scaleLinear: vi.fn(() => ({ domain: vi.fn(), range: vi.fn() })),
  extent: vi.fn(),
  max: vi.fn(),
  min: vi.fn()
}))

// Mock HTMX
Object.defineProperty(window, 'htmx', {
  writable: true,
  value: {
    process: vi.fn(),
    trigger: vi.fn(),
    ajax: vi.fn(),
    swap: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    find: vi.fn(),
    findAll: vi.fn(),
    closest: vi.fn(),
    values: vi.fn(() => ({})),
    remove: vi.fn(),
    addClass: vi.fn(),
    removeClass: vi.fn(),
    toggleClass: vi.fn()
  }
})

// Mock Alpine.js
Object.defineProperty(window, 'Alpine', {
  writable: true,
  value: {
    data: vi.fn(),
    store: vi.fn(),
    directive: vi.fn(),
    magic: vi.fn(),
    plugin: vi.fn(),
    start: vi.fn(),
    clone: vi.fn(),
    nextTick: vi.fn(),
    evaluate: vi.fn(),
    initTree: vi.fn(),
    destroyTree: vi.fn()
  }
})

// Mock fetch
global.fetch = vi.fn()

// Mock WebSocket
global.WebSocket = vi.fn().mockImplementation(() => ({
  close: vi.fn(),
  send: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: WebSocket.CONNECTING,
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3
}))

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn()
  },
  writable: true
})

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn()
  },
  writable: true
})

// Mock performance API
Object.defineProperty(window, 'performance', {
  value: {
    now: vi.fn(() => Date.now()),
    mark: vi.fn(),
    measure: vi.fn(),
    getEntriesByType: vi.fn(() => []),
    getEntriesByName: vi.fn(() => []),
    clearMarks: vi.fn(),
    clearMeasures: vi.fn()
  },
  writable: true
})

// Mock URL
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'blob:url'),
    revokeObjectURL: vi.fn()
  },
  writable: true
})

// Mock navigator
Object.defineProperty(window, 'navigator', {
  value: {
    userAgent: 'test-agent',
    language: 'en-US',
    onLine: true,
    clipboard: {
      writeText: vi.fn(),
      readText: vi.fn()
    },
    serviceWorker: {
      register: vi.fn(),
      ready: Promise.resolve({
        unregister: vi.fn()
      })
    }
  },
  writable: true
})

// Clean up after each test
beforeEach(() => {
  document.body.innerHTML = ''
  vi.clearAllMocks()
})

afterEach(() => {
  document.body.innerHTML = ''
  vi.clearAllTimers()
  vi.restoreAllMocks()
})

// Global test utilities
export const createMockElement = (tag: string = 'div', attributes: Record<string, string> = {}) => {
  const element = document.createElement(tag)
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, value)
  })
  return element
}

export const createMockEvent = (type: string, properties: Record<string, any> = {}) => {
  const event = new Event(type, { bubbles: true, cancelable: true })
  Object.assign(event, properties)
  return event
}

export const waitFor = (callback: () => void | boolean, timeout: number = 1000) => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now()
    const check = () => {
      try {
        const result = callback()
        if (result !== false) {
          resolve(result)
          return
        }
      } catch (error) {
        // Continue checking
      }

      if (Date.now() - startTime >= timeout) {
        reject(new Error('Timeout waiting for condition'))
        return
      }

      setTimeout(check, 10)
    }
    check()
  })
}
