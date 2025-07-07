/**
 * Simple event emitter implementation
 */

export type EventListener<T = any> = (data: T) => void;

export class EventEmitter {
  private listeners: Map<string, Set<EventListener>> = new Map();

  /**
   * Add event listener
   */
  on<T = any>(event: string, listener: EventListener<T>): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
  }

  /**
   * Add one-time event listener
   */
  once<T = any>(event: string, listener: EventListener<T>): void {
    const onceWrapper = (data: T) => {
      listener(data);
      this.off(event, onceWrapper);
    };
    this.on(event, onceWrapper);
  }

  /**
   * Remove event listener
   */
  off<T = any>(event: string, listener: EventListener<T>): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Remove all listeners for an event
   */
  removeAllListeners(event?: string): void {
    if (event) {
      this.listeners.delete(event);
    } else {
      this.listeners.clear();
    }
  }

  /**
   * Emit event to all listeners
   */
  emit<T = any>(event: string, data?: T): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          // Emit error event if listener throws
          setTimeout(() => this.emit('error', error), 0);
        }
      });
    }
  }

  /**
   * Get number of listeners for an event
   */
  listenerCount(event: string): number {
    return this.listeners.get(event)?.size ?? 0;
  }

  /**
   * Get all event names
   */
  eventNames(): string[] {
    return Array.from(this.listeners.keys());
  }

  /**
   * Get listeners for an event
   */
  getListeners(event: string): EventListener[] {
    return Array.from(this.listeners.get(event) ?? []);
  }
}