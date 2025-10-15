import { AnalysisResult, AnalysisResponse } from "@/types/analysis";

export function downloadJSON(obj: any, filename = "results.json") {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(obj, null, 2));
  const link = document.createElement("a");
  link.setAttribute("href", dataStr);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function downloadCSV(results: AnalysisResult[], filename = "results.csv") {
  if (!results || results.length === 0) return;
  const headers = ["filename", "decision", "offset", "confidence", "raw_offset", "sanity_warning", "tags"];
  const rows = results.map(r => [
    r.filename,
    r.decision,
    String(r.offset),
    String(r.confidence),
    r.raw_offset == null ? "" : String(r.raw_offset),
    r.sanity_warning ?? "",
    (r.tags || []).join(";")
  ]);
  const csv = [headers, ...rows].map(r => r.map(cell => {
    // simple escaping
    const cellStr = String(cell ?? "");
    if (cellStr.includes(",") || cellStr.includes('"')) {
      return `"${cellStr.replace(/"/g, '""')}"`;
    }
    return cellStr;
  }).join(",")).join("\n");

  const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
  const link = document.createElement("a");
  link.setAttribute("href", dataStr);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
}
