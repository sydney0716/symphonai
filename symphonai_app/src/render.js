export function element(document, tagName, { className = "", text = "" } = {}) {
  const value = document.createElement(tagName);
  value.className = className;
  value.textContent = text;
  return value;
}

export function append(parent, ...children) {
  parent.append(...children);
  return parent;
}

export function replace(parent, ...children) {
  parent.replaceChildren(...children);
  return parent;
}

export function listen(value, type, listener) {
  value.addEventListener(type, listener);
  return value;
}
