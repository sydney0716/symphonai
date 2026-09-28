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
  showEfforts = false,
  onChoose,
  onCancel,
  onAdjust = () => {},
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

  function detail(row) {
    if (!showEfforts) return "";
    if (row.efforts.length === 0) return "Effort not supported";
    return `Effort: ${row.efforts[row.effortIndex]} · ← → to adjust`;
  }

  function renderRows() {
    const buttons = rows.map((row, index) => {
      const button = element(document, "button", {
        className: `picker-row${index === focused ? " focused" : ""}${row.current ? " current" : ""}`,
      });
      button.type = "button";
      button.setAttribute?.("role", "option");
      button.setAttribute?.("aria-selected", String(index === focused));
      append(
        button,
        element(document, "span", { className: "picker-row-label", text: row.label }),
        element(document, "span", { className: "picker-row-description", text: row.current ? "Current" : "" }),
        element(document, "span", { className: "picker-row-detail", text: detail(row) }),
      );
      listen(button, "click", () => onChoose(row, index));
      return button;
    });
    replace(list, ...buttons);
  }

  function move(amount) {
    if (rows.length === 0) return;
    focused = (focused + amount + rows.length) % rows.length;
    renderRows();
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
    } else if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && focused >= 0 && showEfforts) {
      const row = rows[focused];
      if (row.efforts.length > 0) {
        event.preventDefault();
        row.effortIndex = (row.effortIndex + (event.key === "ArrowRight" ? 1 : -1) + row.efforts.length) % row.efforts.length;
        renderRows();
        onAdjust(row, focused);
      }
    }
  });

  renderRows();
  append(root, heading, list, status);
  root.focus?.();
  return root;
}
