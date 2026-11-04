import { append, element, listen, replace } from "./render.js";
import { lookup } from "./keys.js";

export function createPicker({
  document,
  keymap,
  platform,
  title,
  rows = [],
  initialIndex = 0,
  message = "",
  onChoose,
  onCancel,
}) {
  const root = element(document, "section", { className: "picker" });
  root.tabIndex = 0;
  root.setAttribute?.("role", "dialog");
  root.setAttribute?.("aria-label", title);
  const heading = element(document, "p", { className: "picker-title", text: title });
  const list = element(document, "div", { className: "picker-list" });
  list.setAttribute?.("role", "listbox");
  const status = element(document, "p", { className: "picker-message", text: message });
  let focused = rows.length === 0 ? -1 : Math.max(0, Math.min(initialIndex, rows.length - 1));

  const buttons = rows.map((row, index) => {
    const button = element(document, "button");
    button.type = "button";
    button.setAttribute?.("role", "option");
    append(button, element(document, "span", { className: "picker-row-label", text: row.label }),
      element(document, "span", { className: "picker-row-description", text: row.current ? "Current" : "" }));
    listen(button, "click", () => onChoose(row, index));
    listen(button, "focus", () => { focused = index; renderRows(); });
    return button;
  });
  replace(list, ...buttons);
  function renderRows() {
    buttons.forEach((button, index) => {
      button.className = `picker-row${index === focused ? " focused" : ""}${rows[index].current ? " current" : ""}`;
      button.tabIndex = index === focused ? 0 : -1;
      button.setAttribute?.("aria-selected", String(index === focused));
    });
  }

  function move(amount) {
    if (rows.length === 0) return;
    focused = (focused + amount + rows.length) % rows.length;
    renderRows();
    buttons[focused].focus();
  }

  listen(root, "keydown", (event) => {
    if (lookup(keymap, event, { platform }) === "cancel") {
      event.preventDefault();
      onCancel();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      move(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      move(-1);
    } else if (event.key === "Enter" && focused >= 0) {
      event.preventDefault();
      onChoose(rows[focused], focused);
    }
  });

  renderRows();
  append(root, heading, list, status);
  return root;
}
