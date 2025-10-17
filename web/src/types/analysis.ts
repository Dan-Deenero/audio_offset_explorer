export type Decision = "green" | "yellow" | "red";

export type AnalysisResult = {
  filename: string;
  decision: Decision;
  offset: number;
  confidence: number;
  sanity_warning?: string | null;
  raw_offset?: number;
  tags?: string[];
  aligned_audio_rel?: string | null; // ✅ already here
};

export type AnalysisResponse = {
  reference: string;
  generatedAt: string;
  total: number;
  results: AnalysisResult[];
  logs?: string[];
};
