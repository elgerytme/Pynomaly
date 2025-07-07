// HTMX Extensions for Enhanced Functionality
if (typeof htmx !== "undefined") {
  // Loading indicator extension
  htmx.defineExtension("loading-states", {
    onEvent: function (name, evt) {
      if (name === "htmx:beforeRequest") {
        evt.target.classList.add("htmx-loading");
      } else if (name === "htmx:afterRequest") {
        evt.target.classList.remove("htmx-loading");
      }
    },
  });

  // Auto-retry extension for failed requests
  htmx.defineExtension("auto-retry", {
    onEvent: function (name, evt) {
      if (name === "htmx:responseError") {
        const retryCount = parseInt(evt.target.dataset.retryCount || "0");
        const maxRetries = parseInt(evt.target.dataset.maxRetries || "3");

        if (retryCount < maxRetries) {
          evt.target.dataset.retryCount = (retryCount + 1).toString();
          setTimeout(
            () => {
              htmx.trigger(evt.target, "click");
            },
            1000 * Math.pow(2, retryCount),
          ); // Exponential backoff
        }
      }
    },
  });
}
