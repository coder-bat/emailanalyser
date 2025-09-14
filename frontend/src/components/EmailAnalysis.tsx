import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tab,
  Tabs,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  Alert,
  CircularProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import { EmailMessage, SenderStats, ActionableSender } from '../types';
import { emailAnalysisApi } from '../utils/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EmailAnalysis: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [senderStats, setSenderStats] = useState<SenderStats[]>([]);
  const [sendersToDelete, setSendersToDelete] = useState<ActionableSender[]>([]);
  const [importantSenders, setImportantSenders] = useState<ActionableSender[]>([]);
  
  const [emailsPage, setEmailsPage] = useState(0);
  const [emailsRowsPerPage, setEmailsRowsPerPage] = useState(10);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [runAnalysisOpen, setRunAnalysisOpen] = useState(false);
  const [analysisConfig, setAnalysisConfig] = useState({
    email: '',
    max_emails: 1000,
    categories: '',
    unread_only: false,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [emailsData, senderStatsData, deleteData, importantData] = await Promise.all([
          emailAnalysisApi.getEmails(100),
          emailAnalysisApi.getSenderStats(),
          emailAnalysisApi.getSendersToDelete(),
          emailAnalysisApi.getImportantSenders(),
        ]);
        
        setEmails(emailsData);
        setSenderStats(senderStatsData);
        setSendersToDelete(deleteData);
        setImportantSenders(importantData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleEmailsPageChange = (event: unknown, newPage: number) => {
    setEmailsPage(newPage);
  };

  const handleEmailsRowsPerPageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEmailsRowsPerPage(parseInt(event.target.value, 10));
    setEmailsPage(0);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getImportanceColor = (score: number) => {
    if (score >= 0.7) return 'error';
    if (score >= 0.4) return 'warning';
    return 'default';
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, any> = {
      promotional: 'secondary',
      work: 'primary',
      personal: 'success',
      newsletter: 'info',
      spam: 'error',
      other: 'default',
    };
    return colors[category] || 'default';
  };

  const handleRunAnalysis = async () => {
    try {
      await emailAnalysisApi.runAnalysis(analysisConfig);
      setRunAnalysisOpen(false);
      // Refresh data after analysis
      window.location.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run analysis');
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4" gutterBottom>
          Email Analysis Details
        </Typography>
        <Button
          variant="contained"
          onClick={() => setRunAnalysisOpen(true)}
        >
          Run New Analysis
        </Button>
      </Box>

      <Paper sx={{ width: '100%' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab label="Email Data" />
          <Tab label="Sender Statistics" />
          <Tab label="Senders to Delete" />
          <Tab label="Important Senders" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>From</TableCell>
                  <TableCell>Subject</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Importance</TableCell>
                  <TableCell>Attachments</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {emails
                  .slice(emailsPage * emailsRowsPerPage, emailsPage * emailsRowsPerPage + emailsRowsPerPage)
                  .map((email) => (
                    <TableRow key={email.uid}>
                      <TableCell>{formatDate(email.date)}</TableCell>
                      <TableCell>{email.sender}</TableCell>
                      <TableCell>{email.subject}</TableCell>
                      <TableCell>
                        <Chip
                          label={email.category}
                          color={getCategoryColor(email.category)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={email.importance_score.toFixed(2)}
                          color={getImportanceColor(email.importance_score)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{email.has_attachments ? 'Yes' : 'No'}</TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={emails.length}
            rowsPerPage={emailsRowsPerPage}
            page={emailsPage}
            onPageChange={handleEmailsPageChange}
            onRowsPerPageChange={handleEmailsRowsPerPageChange}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Sender</TableCell>
                  <TableCell>Total Emails</TableCell>
                  <TableCell>Important Emails</TableCell>
                  <TableCell>Importance Rate</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {senderStats.map((sender) => (
                  <TableRow key={sender.sender_key}>
                    <TableCell>{sender.sender_label}</TableCell>
                    <TableCell>{sender.total_emails}</TableCell>
                    <TableCell>{sender.important_emails}</TableCell>
                    <TableCell>
                      {((sender.important_emails / sender.total_emails) * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Recommended for Cleanup
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Sender</TableCell>
                  <TableCell>Email Count</TableCell>
                  <TableCell>Avg Importance</TableCell>
                  <TableCell>Newsletter %</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sendersToDelete.map((sender, index) => (
                  <TableRow key={index}>
                    <TableCell>{sender.sender}</TableCell>
                    <TableCell>{sender.count}</TableCell>
                    <TableCell>{sender.avg_importance.toFixed(2)}</TableCell>
                    <TableCell>{((sender.newsletter_pct || 0) * 100).toFixed(1)}%</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Typography variant="h6" gutterBottom>
            High Priority Senders
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Sender</TableCell>
                  <TableCell>Email Count</TableCell>
                  <TableCell>Avg Importance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {importantSenders.map((sender, index) => (
                  <TableRow key={index}>
                    <TableCell>{sender.sender}</TableCell>
                    <TableCell>{sender.count}</TableCell>
                    <TableCell>{sender.avg_importance.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </Paper>

      {/* Run Analysis Dialog */}
      <Dialog open={runAnalysisOpen} onClose={() => setRunAnalysisOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Run Email Analysis</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Email Address"
            type="email"
            fullWidth
            variant="outlined"
            value={analysisConfig.email}
            onChange={(e) => setAnalysisConfig({ ...analysisConfig, email: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Max Emails"
            type="number"
            fullWidth
            variant="outlined"
            value={analysisConfig.max_emails}
            onChange={(e) => setAnalysisConfig({ ...analysisConfig, max_emails: parseInt(e.target.value) })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Gmail Categories (e.g., Primary,Promotions)"
            fullWidth
            variant="outlined"
            value={analysisConfig.categories}
            onChange={(e) => setAnalysisConfig({ ...analysisConfig, categories: e.target.value })}
            sx={{ mb: 2 }}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={analysisConfig.unread_only}
                onChange={(e) => setAnalysisConfig({ ...analysisConfig, unread_only: e.target.checked })}
              />
            }
            label="Unread emails only"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRunAnalysisOpen(false)}>Cancel</Button>
          <Button onClick={handleRunAnalysis} variant="contained">Run Analysis</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default EmailAnalysis;