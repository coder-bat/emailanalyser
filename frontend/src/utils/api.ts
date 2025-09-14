import axios from 'axios';
import { EmailMessage, SenderStats, ActionableSender, AnalysisPatterns, AnalysisSummary } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const emailAnalysisApi = {
  // Get analysis summary
  getSummary: async (): Promise<AnalysisSummary> => {
    const response = await api.get('/api/summary');
    return response.data;
  },

  // Get email data
  getEmails: async (limit?: number): Promise<EmailMessage[]> => {
    const response = await api.get('/api/emails', { params: { limit } });
    return response.data;
  },

  // Get sender statistics
  getSenderStats: async (): Promise<SenderStats[]> => {
    const response = await api.get('/api/sender-stats');
    return response.data;
  },

  // Get senders to delete
  getSendersToDelete: async (): Promise<ActionableSender[]> => {
    const response = await api.get('/api/senders-to-delete');
    return response.data;
  },

  // Get important senders
  getImportantSenders: async (): Promise<ActionableSender[]> => {
    const response = await api.get('/api/important-senders');
    return response.data;
  },

  // Get analysis patterns
  getPatterns: async (): Promise<AnalysisPatterns> => {
    const response = await api.get('/api/patterns');
    return response.data;
  },

  // Run email analysis
  runAnalysis: async (config: {
    email?: string;
    max_emails?: number;
    categories?: string;
    unread_only?: boolean;
  }): Promise<{ message: string; job_id?: string }> => {
    const response = await api.post('/api/run-analysis', config);
    return response.data;
  },

  // Get analysis status
  getAnalysisStatus: async (jobId: string): Promise<{ status: string; progress?: number; error?: string }> => {
    const response = await api.get(`/api/analysis-status/${jobId}`);
    return response.data;
  },
};

export default emailAnalysisApi;