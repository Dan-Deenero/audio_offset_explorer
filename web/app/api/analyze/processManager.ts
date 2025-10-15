import type { ChildProcessWithoutNullStreams } from "child_process";

let currentProcess: ChildProcessWithoutNullStreams | null = null;
let wasStoppedByUser = false;

export function setProcess(proc: ChildProcessWithoutNullStreams) {
  currentProcess = proc;
  wasStoppedByUser = false; // 👈 reset flag whenever a new process starts
}

export function clearProcess() {
  currentProcess = null;
  wasStoppedByUser = false;
}

export function stopProcess(): boolean {
  if (currentProcess) {
    wasStoppedByUser = true;
    currentProcess.kill("SIGKILL");
    currentProcess = null;
    return true;
  }
  return false;
}

export function getWasStoppedByUser(): boolean {
  return wasStoppedByUser;
}
