import "@testing-library/jest-dom";

// jsdom does not implement scrollIntoView; stub it for components that
// auto-scroll (e.g. the Hearth chat area).
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
