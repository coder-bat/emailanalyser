export interface EmailMessage {
  uid: string;
  subject: string;
  sender: string;
  sender_email?: string;
  recipient: string;
  date: string;
  body: string;
  category: string;
  importance_score: number;
  has_attachments: boolean;
  size?: number;
}

export interface SenderStats {
  sender_key: string;
  sender_label: string;
  total_emails: number;
  important_emails: number;
}

export interface ActionableSender {
  sender: string;
  count: number;
  avg_importance: number;
  newsletter_pct?: number;
}

export interface AnalysisPatterns {
  time_patterns: {
    peak_hours: [number, number][];
    peak_days: [string, number][];
    hour_distribution: Record<number, number>;
    day_distribution: Record<string, number>;
  };
  sender_patterns: {
    top_senders: [string, number][];
    sender_categories: Record<string, string>;
  };
  category_distribution: Record<string, number>;
  attachment_patterns: {
    percentage_with_attachments: number;
    common_types: [string, number][];
  };
}

export interface AnalysisSummary {
  total_senders: number;
  senders_to_delete_count: number;
  important_senders_count: number;
  total_emails: number;
  important_emails: number;
  date_range: {
    start: string;
    end: string;
  };
}