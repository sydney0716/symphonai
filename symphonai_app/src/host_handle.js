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


export function resolveHost(env) {
  if (env?.__symphonaiShell !== undefined) {
    return validatedHost(env.__symphonaiShell, "shell bridge");
  }
  if (env?.__symphonai !== undefined) {
    return validatedHost(env.__symphonai, "page handshake");
  }
  throw new HostError("no host handshake is available");
}
