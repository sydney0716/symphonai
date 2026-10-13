export const PAGES = Object.freeze(["chat", "settings"]);
export const DEFAULT_ROUTE = Object.freeze({ page: "chat", section: "" });

const MEMBER = /^[a-z][a-z0-9-]*$/;

export function parseRoute(fragment) {
  if (typeof fragment !== "string" || !fragment.startsWith("#/")) {
    return DEFAULT_ROUTE;
  }
  const members = fragment.slice(2).split("/");
  if (
    members.length < 1
    || members.length > 2
    || !PAGES.includes(members[0])
    || members.some((member) => !MEMBER.test(member))
  ) {
    return DEFAULT_ROUTE;
  }
  return { page: members[0], section: members[1] ?? "" };
}

export function formatRoute(route) {
  if (
    !route
    || !PAGES.includes(route.page)
    || typeof route.section !== "string"
    || (route.section !== "" && !MEMBER.test(route.section))
  ) {
    return "#/chat";
  }
  return `#/${route.page}${route.section ? `/${route.section}` : ""}`;
}
