export class HostError extends Error {
  constructor(message) {
    super(message);
    this.name = "HostError";
  }
}


function validatedHost(value, source) {
  if (
    value === null
    || typeof value !== "object"
    || !Number.isInteger(value.port)
    || value.port < 1
    || value.port > 65535
    || typeof value.token !== "string"
    || value.token.length === 0
  ) {
    throw new HostError(`${source} did not provide a valid port and token`);
  }
  return { port: value.port, token: value.token };
}


export function hostFromShell(bridge) {
  return validatedHost(bridge, "shell bridge");
}


export function hostFromPage(global) {
  return validatedHost(global?.__symphonai, "page handshake");
}


export function resolveHost(env) {
  if (env?.__symphonaiShell !== undefined) {
    return hostFromShell(env.__symphonaiShell);
  }
  if (env?.__symphonai !== undefined) {
    return hostFromPage(env);
  }
  throw new HostError("no host handshake is available");
}
