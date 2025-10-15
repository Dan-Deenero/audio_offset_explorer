export type Decision = "green" | "yellow" | "red";

export type AnalysisResult = {
  filename: string;
  decision: Decision;
  offset: number;           // final offset (seconds)
  confidence: number;       // 0-100
  sanity_warning?: string | null;
  raw_offset?: number;
  tags?: string[];
};

export type AnalysisResponse = {
  reference: string;
  generatedAt: string;
  total: number;
  results: AnalysisResult[];
  logs?: string[];
};
