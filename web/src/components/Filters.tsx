import React from "react";

type Props = {
  decision: "all" | "green" | "yellow" | "red";
  setDecision: (d: "all" | "green" | "yellow" | "red") => void;
  minConfidence: number;  
  setMinConfidence: (v: number) => void;
};

export default function Filters({ decision, setDecision, minConfidence, setMinConfidence }: Props) {
  return (
    <div className="p-3 border border-neutral-content rounded bg-white flex items-center gap-4">
      <div className="w-1/4">
        <label className="block text-sm font-medium">Decision</label>
        <select value={decision} onChange={(e) => setDecision(e.target.value as any)} className="select select-lg">
          <option value="all">All</option>
          <option value="green">Green</option>
          <option value="yellow">Yellow</option>
          <option value="red">Red</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium">Min confidence: {minConfidence}%</label>
        <input
          type="range"
          min={0}
          max={100}
          value={minConfidence}
          onChange={(e) => setMinConfidence(Number(e.target.value))}
          className="mt-2"
        />
      </div>
    </div>
  );
}
