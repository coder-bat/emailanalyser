import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from 'recharts';
import { emailAnalysisApi } from '../utils/api';
import { AnalysisSummary, AnalysisPatterns } from '../types';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const Dashboard: React.FC = () => {
  const [summary, setSummary] = useState<AnalysisSummary | null>(null);
  const [patterns, setPatterns] = useState<AnalysisPatterns | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [summaryData, patternsData] = await Promise.all([
          emailAnalysisApi.getSummary(),
          emailAnalysisApi.getPatterns(),
        ]);
        setSummary(summaryData);
        setPatterns(patternsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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

  // Prepare chart data
  const categoryData = patterns?.category_distribution 
    ? Object.entries(patterns.category_distribution).map(([key, value]) => ({
        name: key.charAt(0).toUpperCase() + key.slice(1),
        value: value,
      }))
    : [];

  const hourData = patterns?.time_patterns?.hour_distribution
    ? Object.entries(patterns.time_patterns.hour_distribution).map(([hour, count]) => ({
        hour: `${hour}:00`,
        emails: count,
      }))
    : [];

  const topSendersData = patterns?.sender_patterns?.top_senders?.slice(0, 10).map(([sender, count]) => ({
    sender: sender.length > 30 ? sender.substring(0, 30) + '...' : sender,
    count,
  })) || [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Email Analysis Dashboard
      </Typography>

      {/* Summary Cards */}
      <Box display="flex" flexWrap="wrap" gap={3} sx={{ mb: 4 }}>
        <Box flex="1 1 300px" minWidth="250px">
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Emails
              </Typography>
              <Typography variant="h4">
                {summary?.total_emails || 0}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box flex="1 1 300px" minWidth="250px">
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Important Emails
              </Typography>
              <Typography variant="h4">
                {summary?.important_emails || 0}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box flex="1 1 300px" minWidth="250px">
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Senders
              </Typography>
              <Typography variant="h4">
                {summary?.total_senders || 0}
              </Typography>
            </CardContent>
          </Card>
        </Box>
        <Box flex="1 1 300px" minWidth="250px">
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Actionable Senders
              </Typography>
              <Typography variant="h4">
                {(summary?.senders_to_delete_count || 0) + (summary?.important_senders_count || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Charts */}
      <Box display="flex" flexDirection="column" gap={3}>
        <Box display="flex" flexWrap="wrap" gap={3}>
          {/* Email Categories */}
          <Box flex="1 1 500px" minWidth="400px">
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Email Categories
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={categoryData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry: any) => `${entry.name} ${(entry.percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Box>

          {/* Email by Hour */}
          <Box flex="1 1 500px" minWidth="400px">
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Emails by Hour
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hourData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="emails" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Box>
        </Box>

        {/* Top Senders */}
        <Box>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Top Senders
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topSendersData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="sender" type="category" width={200} />
                <Tooltip />
                <Bar dataKey="count" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default Dashboard;