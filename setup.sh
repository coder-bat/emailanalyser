#!/bin/bash
set -e

echo "🚀 Starting EmailAnalyser setup..."

# Create output directory and sample data
mkdir -p email_analysis_output

# Generate sample email data CSV
cat > email_analysis_output/email_data_sample.csv << 'EOF'
date,sender,sender_email,subject,category,importance_score,has_attachments
2024-01-15T10:30:00,John Doe <john@company.com>,john@company.com,Project Update Meeting,work,0.85,false
2024-01-15T09:15:00,Newsletter <news@newsletter.com>,news@newsletter.com,Weekly Tech Digest,newsletter,0.25,false
2024-01-14T14:22:00,Sales Team <sales@shop.com>,sales@shop.com,50% Off Everything!,promotional,0.15,false
2024-01-14T16:45:00,Mom <mom@family.com>,mom@family.com,Family Dinner Plans,personal,0.70,true
2024-01-13T11:30:00,Boss <boss@company.com>,boss@company.com,Urgent: Budget Review,work,0.95,true
2024-01-13T08:20:00,Spam <winner@spam.com>,winner@spam.com,You Won $1000000!,spam,0.05,false
2024-01-12T15:10:00,Support <support@service.com>,support@service.com,Ticket Update #12345,work,0.60,false
2024-01-12T12:00:00,Friend <friend@email.com>,friend@email.com,Coffee Plans?,personal,0.55,false
2024-01-11T09:30:00,Finance <finance@company.com>,finance@company.com,Monthly Report Ready,work,0.80,true
2024-01-11T16:45:00,Store <promo@store.com>,promo@store.com,Flash Sale Tonight,promotional,0.20,false
EOF

# Generate sender stats CSV
cat > email_analysis_output/sender_stats.csv << 'EOF'
sender_key,sender_label,total_emails,important_emails
john@company.com,John Doe <john@company.com>,15,8
boss@company.com,Boss <boss@company.com>,8,7
mom@family.com,Mom <mom@family.com>,12,6
finance@company.com,Finance <finance@company.com>,6,5
support@service.com,Support <support@service.com>,10,4
news@newsletter.com,Newsletter <news@newsletter.com>,25,2
friend@email.com,Friend <friend@email.com>,8,3
sales@shop.com,Sales Team <sales@shop.com>,20,1
promo@store.com,Store <promo@store.com>,15,1
winner@spam.com,Spam <winner@spam.com>,5,0
EOF

# Generate senders to delete CSV
cat > email_analysis_output/senders_to_delete.csv << 'EOF'
sender,count,avg_importance,newsletter_pct
sales@shop.com,20,0.18,0.85
promo@store.com,15,0.16,0.90
winner@spam.com,5,0.05,0.20
news@newsletter.com,25,0.25,1.00
EOF

# Generate important senders CSV
cat > email_analysis_output/important_senders.csv << 'EOF'
sender,count,avg_importance
boss@company.com,8,0.87
finance@company.com,6,0.78
john@company.com,15,0.72
mom@family.com,12,0.68
support@service.com,10,0.62
EOF

# Generate summary JSON
cat > email_analysis_output/summary.json << 'EOF'
{
  "total_senders": 10,
  "senders_to_delete_count": 4,
  "important_senders_count": 5
}
EOF

echo "✅ Sample data created in email_analysis_output/"

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    pip install flask flask-cors gunicorn
fi

# Install and build frontend if it exists
if [ -d "frontend" ]; then
    echo "🎨 Building frontend..."
    cd frontend
    npm install
    npm run build
    cd ..
fi

echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Run API server: python api_server.py"
echo "  2. Visit: http://localhost:5000"
echo ""
echo "To use Docker:"
echo "  docker-compose up -d"