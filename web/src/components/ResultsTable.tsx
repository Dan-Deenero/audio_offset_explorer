import React, { useState } from "react";
import { AnalysisResult } from "@/types/analysis";

type Props = {
  results: AnalysisResult[];
  referenceFile?: File | null;
  onPlayCandidate?: (blobUrl: string) => void;
};

export default function ResultsTable({ results, referenceFile, onPlayCandidate }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);
  const audioRef = React.useRef<HTMLAudioElement | null>(null);

  const toggle = (name: string) => setExpanded((s) => ({ ...s, [name]: !s[name] }));

  const playLocalCandidate = (fileName: string) => {
    // if the candidate was uploaded but is only local in the browser page, we cannot reference it by name;
    // the page.tsx provides a play callback for local preview. If not present, ignore.
  };

  return (
    <div className="overflow-x-auto rounded-box border border-base-content/5 bg-base-100">
      <table className="w-full text-left table">
        <thead className="bg-gray-50">
          <tr>
            <th className="p-2">Filename</th>
            <th className="p-2">Decision</th>
            <th className="p-2">Offset (s)</th>
            <th className="p-2">Confidence</th>
            <th className="p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <React.Fragment key={r.filename}>
              <tr className="border-t hover:bg-gray-50">
                <td className="p-2">{r.filename}</td>
                <td className="p-2">
                  <span
                    className={`px-2 py-1 rounded text-sm ${
                      r.decision === "green" ? "bg-green-100 text-green-700" : r.decision === "yellow" ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700"
                    }`}
                  >
                    {r.decision}
                  </span>
                </td>
                <td className="p-2">{r.offset}</td>
                <td className="p-2">{r.confidence.toFixed(1)}%</td>
                <td className="p-2">
                  <button className="mr-2 text-sm underline" onClick={() => toggle(r.filename)}>
                    {expanded[r.filename] ? "Hide" : "Details"}
                  </button>
                  {onPlayCandidate && (
                    <button
                      className="text-sm border px-2 py-1 rounded"
                      onClick={() => {
                        // Backend doesn't return candidate file URLs — frontend local preview (if present) handled at Page level.
                        // If you host uploaded candidates and return URLs, you can play them via <audio src=...>.
                        alert("Local preview available before upload. After analysis, implement candidate streaming if desired.");
                      }}
                    >
                      Preview
                    </button>
                  )}
                </td>
              </tr>

              {expanded[r.filename] && (
                <tr className="bg-gray-50">
                  <td colSpan={5} className="p-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-sm text-gray-600">Sanity warning</div>
                        <div>{r.sanity_warning ?? "—"}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Raw offset</div>
                        <div>{r.raw_offset ?? "—"}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Final offset</div>
                        <div>{r.offset}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Tags</div>
                        <div>{(r.tags || []).join(", ") || "—"}</div>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}

          {results.length === 0 && (
            <tr>
              <td colSpan={5} className="p-4 text-gray-500">No results to display.</td>
            </tr>
          )}
        </tbody>
      </table>

      <audio ref={audioRef} style={{ display: "none" }} />
    </div>
  );
}
