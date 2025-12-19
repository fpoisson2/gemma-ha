export interface DatasetExample {
  id: number;
  text: string;
  parsed: ParsedExample;
  status: 'pending' | 'approved' | 'rejected';
  rejectionReason?: string;
}

export interface ParsedExample {
  userQuery: string;
  entities: EntityList[];
  modelResponse: string;
  functionCall?: FunctionCall;
}

export interface EntityList {
  domain: string;
  entities: string[];
}

export interface FunctionCall {
  action: string;
  entityId: string;
}

export interface CleaningDecision {
  id: number;
  originalText: string;
  status: 'approved' | 'rejected';
  rejectionReason?: string;
  timestamp: string;
}

export interface CleaningReport {
  exportDate: string;
  totalExamples: number;
  approved: number;
  rejected: number;
  pending: number;
  decisions: CleaningDecision[];
}
