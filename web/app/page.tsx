"use client";

import React, { useState } from "react";
import { AnalysisResponse, AnalysisResult } from "@/types/analysis";
import ResultsTable from "@/components/ResultsTable";
import FileUpload from "@/components/FileUpload";
import { downloadJSON, downloadCSV } from "@/utils/exports";
import AudioUploader from "@/components/AudioUploader";

export default function Page() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [candidateFiles, setCandidateFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const [filterDecision, setFilterDecision] = useState<"all" | "green" | "yellow" | "red">("all");
  const [minConfidence, setMinConfidence] = useState<number>(0);

  const runAnalysis = async () => {
  setError(null);
  if (!referenceFile || candidateFiles.length === 0) {
    setError("Please select a reference and at least one candidate file.");
    return;
  }

  setLoading(true);
  setResponse(null);
  setLogs([]);

  // 🔸 Start listening to live logs before uploading
  const eventSource = new EventSource("/api/analyze/stream");
  eventSource.onmessage = (event) => {
    setLogs((prev) => [...prev, event.data]);
  };
  eventSource.onerror = () => {
    console.warn("Log stream error");
    eventSource.close();
  };

  try {
    const form = new FormData();
    form.append("reference", referenceFile);
    candidateFiles.forEach((file) => form.append("candidates", file));

    const res = await fetch("/api/analyze", {
      method: "POST",
      body: form,
    });

    const json = await res.json();
    eventSource.close();

    if (!res.ok) {
      setError(json?.error || "Server error");
      if (json?.logs) setLogs(json.logs);
      return;
    }

    setResponse(json as AnalysisResponse);
    if (json.logs) setLogs((prev) => [...prev, ...json.logs]);
  } catch (e: any) {
    setError(e?.message || "Network error");
  } finally {
    setLoading(false);
  }
};


  const resetAll = () => {
    setReferenceFile(null);
    setCandidateFiles([]);
    setResponse(null);
    setError(null);
    setLogs([]);
  };

  const filteredResults = (response?.results ?? []).filter((r: AnalysisResult) => {
    if (filterDecision !== "all" && r.decision !== filterDecision) return false;
    if (r.confidence < minConfidence) return false;
    return true;
  });

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-semibold">Audio Offset Explorer</h1>

      <AudioUploader referenceFile={referenceFile}
        setReferenceFile={setReferenceFile}
        candidateFiles={candidateFiles}
        setCandidateFiles={setCandidateFiles} />

      <div className="flex justify-center gap-3">
        {loading ? (
          <button className="btn rounded" disabled>
            <span className="loading loading-spinner"></span>
            Running analysis…
          </button>
        ) : (
          <button className="btn btn-primary text-black rounded" onClick={runAnalysis}>
            Run Analysis
          </button>
        )}
        <button className="btn btn-outline rounded" onClick={resetAll}>
          Reset
        </button>
      </div>

      {response && (
        <div className="text-sm text-gray-600">
          Last run: {new Date(response.generatedAt).toLocaleString()}
        </div>
      )}

      {error && (
        <div className="toast toast-top toast-center">
          <div className="alert alert-error">
            <span>{error}</span>
          </div>
        </div>
      )}

      {response && (
        <section className="mb-6">
          <div className="flex items-center gap-3 mb-3">
            <h2 className="text-lg font-semibold">
              Results ({filteredResults.length}/{response?.total ?? 0})
            </h2>
            <div className="ml-auto flex gap-2">
              <button
                className="px-3 py-1 btn border rounded"
                onClick={() => downloadJSON(response, `${response.reference}_results.json`)}
              >
                Export JSON
              </button>
              <button
                className="px-3 py-1 btn border rounded"
                onClick={() => downloadCSV(response.results, `${response.reference}_results.csv`)}
              >
                Export CSV
              </button>
            </div>
          </div>

          <ResultsTable
            results={filteredResults}
            referenceFile={referenceFile}
            onPlayCandidate={(blobUrl) => {
              const audio = new Audio(blobUrl);
              audio.play().catch(() => { });
            }}
          />
        </section>
      )}

      <section className="mb-6 p-4 border rounded bg-white">
        <h3 className="font-medium mb-2">Server logs (last lines)</h3>
        <div className="bg-slate-50 p-2 rounded h-40 overflow-auto text-xs">
          {logs.length === 0 ? (
            <div className="text-gray-500">No logs yet.</div>
          ) : (
            logs.map((l, i) => <div key={i}>{l}</div>)
          )}
        </div>
      </section>
    </div>
  );
}
