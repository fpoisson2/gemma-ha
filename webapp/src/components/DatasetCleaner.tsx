import { useState, useCallback } from 'react';
import type { DatasetExample, ParsedExample, CleaningReport } from '../types';
import { ExampleCard } from './ExampleCard';
import './DatasetCleaner.css';

export function DatasetCleaner() {
  const [examples, setExamples] = useState<DatasetExample[]>([]);
  const [filter, setFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const parseExample = (text: string): ParsedExample => {
    const result: ParsedExample = {
      userQuery: '',
      entities: [],
      modelResponse: '',
    };

    // Extract user query
    const userMatch = text.match(/<start_of_turn>user\n([\s\S]*?)<end_of_turn>/);
    if (userMatch) {
      const userContent = userMatch[1];

      // Separate query from entities
      const lines = userContent.split('\n');
      const queryLines: string[] = [];
      const entityLines: string[] = [];

      for (const line of lines) {
        if (line.startsWith('Entités ')) {
          entityLines.push(line);
        } else if (entityLines.length === 0) {
          queryLines.push(line);
        }
      }

      result.userQuery = queryLines.join('\n').trim();

      // Parse entities
      for (const entityLine of entityLines) {
        const domainMatch = entityLine.match(/Entités (\w+) disponibles: (.+)/);
        if (domainMatch) {
          const domain = domainMatch[1];
          const entities = domainMatch[2].split(', ').map(e => e.trim());
          result.entities.push({ domain, entities });
        }
      }
    }

    // Extract model response
    const modelMatch = text.match(/<start_of_turn>model\n([\s\S]*?)(?:<end_of_turn>|$)/);
    if (modelMatch) {
      result.modelResponse = modelMatch[1].trim();

      // Try to parse function call
      const funcMatch = result.modelResponse.match(/<start_function_call>call:(\w+\.\w+)\{entity_id:<escape>([^<]+)<escape>/);
      if (funcMatch) {
        result.functionCall = {
          action: funcMatch[1],
          entityId: funcMatch[2],
        };
      }
    }

    return result;
  };

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    const reader = new FileReader();

    reader.onload = (e) => {
      const content = e.target?.result as string;
      const lines = content.split('\n').filter(line => line.trim());

      const parsed: DatasetExample[] = lines.map((line, index) => {
        try {
          const json = JSON.parse(line);
          return {
            id: index + 1,
            text: json.text || '',
            parsed: parseExample(json.text || ''),
            status: 'pending' as const,
          };
        } catch {
          return {
            id: index + 1,
            text: line,
            parsed: { userQuery: 'Erreur de parsing', entities: [], modelResponse: '' },
            status: 'pending' as const,
          };
        }
      });

      setExamples(parsed);
      setIsLoading(false);
    };

    reader.readAsText(file);
  }, []);

  const handleApprove = useCallback((id: number) => {
    setExamples(prev => prev.map(ex =>
      ex.id === id ? { ...ex, status: 'approved' as const, rejectionReason: undefined } : ex
    ));
  }, []);

  const handleReject = useCallback((id: number, reason?: string) => {
    setExamples(prev => prev.map(ex =>
      ex.id === id ? { ...ex, status: 'rejected' as const, rejectionReason: reason } : ex
    ));
  }, []);

  const handleReset = useCallback((id: number) => {
    setExamples(prev => prev.map(ex =>
      ex.id === id ? { ...ex, status: 'pending' as const, rejectionReason: undefined } : ex
    ));
  }, []);

  const exportDecisions = useCallback(() => {
    const report: CleaningReport = {
      exportDate: new Date().toISOString(),
      totalExamples: examples.length,
      approved: examples.filter(e => e.status === 'approved').length,
      rejected: examples.filter(e => e.status === 'rejected').length,
      pending: examples.filter(e => e.status === 'pending').length,
      decisions: examples
        .filter(e => e.status !== 'pending')
        .map(e => ({
          id: e.id,
          originalText: e.text,
          status: e.status as 'approved' | 'rejected',
          rejectionReason: e.rejectionReason,
          timestamp: new Date().toISOString(),
        })),
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dataset-cleaning-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [examples]);

  const exportCleanedDataset = useCallback(() => {
    const cleanedExamples = examples
      .filter(e => e.status !== 'rejected')
      .map(e => ({ text: e.text }));

    const jsonl = cleanedExamples.map(e => JSON.stringify(e)).join('\n');
    const blob = new Blob([jsonl], { type: 'application/jsonl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `train-cleaned-${new Date().toISOString().split('T')[0]}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  }, [examples]);

  const filteredExamples = examples.filter(ex => {
    if (filter !== 'all' && ex.status !== filter) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        ex.parsed.userQuery.toLowerCase().includes(query) ||
        ex.parsed.modelResponse.toLowerCase().includes(query) ||
        ex.parsed.entities.some(el =>
          el.entities.some(e => e.toLowerCase().includes(query))
        )
      );
    }
    return true;
  });

  const stats = {
    total: examples.length,
    approved: examples.filter(e => e.status === 'approved').length,
    rejected: examples.filter(e => e.status === 'rejected').length,
    pending: examples.filter(e => e.status === 'pending').length,
  };

  const approveAllPending = useCallback(() => {
    setExamples(prev => prev.map(ex =>
      ex.status === 'pending' ? { ...ex, status: 'approved' as const } : ex
    ));
  }, []);

  return (
    <div className="dataset-cleaner">
      <header className="header">
        <h1>🧹 Dataset Cleaner</h1>
        <p>Nettoyage et validation du dataset Home Assistant</p>
      </header>

      {examples.length === 0 ? (
        <div className="upload-section">
          <label className="upload-area">
            <input
              type="file"
              accept=".jsonl"
              onChange={handleFileUpload}
              hidden
            />
            <div className="upload-content">
              <span className="upload-icon">📁</span>
              <span className="upload-text">
                {isLoading ? 'Chargement...' : 'Cliquez pour uploader train.jsonl'}
              </span>
              <span className="upload-hint">ou glissez-déposez le fichier ici</span>
            </div>
          </label>
        </div>
      ) : (
        <>
          <div className="stats-bar">
            <div className="stat">
              <span className="stat-value">{stats.total}</span>
              <span className="stat-label">Total</span>
            </div>
            <div className="stat stat-pending">
              <span className="stat-value">{stats.pending}</span>
              <span className="stat-label">En attente</span>
            </div>
            <div className="stat stat-approved">
              <span className="stat-value">{stats.approved}</span>
              <span className="stat-label">Approuvés</span>
            </div>
            <div className="stat stat-rejected">
              <span className="stat-value">{stats.rejected}</span>
              <span className="stat-label">Rejetés</span>
            </div>
            <div className="stat stat-progress">
              <div className="progress-bar">
                <div
                  className="progress-approved"
                  style={{ width: `${(stats.approved / stats.total) * 100}%` }}
                />
                <div
                  className="progress-rejected"
                  style={{ width: `${(stats.rejected / stats.total) * 100}%` }}
                />
              </div>
              <span className="stat-label">
                {Math.round(((stats.approved + stats.rejected) / stats.total) * 100)}% traité
              </span>
            </div>
          </div>

          <div className="toolbar">
            <div className="filter-group">
              <button
                className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
                onClick={() => setFilter('all')}
              >
                Tous ({stats.total})
              </button>
              <button
                className={`filter-btn ${filter === 'pending' ? 'active' : ''}`}
                onClick={() => setFilter('pending')}
              >
                En attente ({stats.pending})
              </button>
              <button
                className={`filter-btn ${filter === 'approved' ? 'active' : ''}`}
                onClick={() => setFilter('approved')}
              >
                Approuvés ({stats.approved})
              </button>
              <button
                className={`filter-btn ${filter === 'rejected' ? 'active' : ''}`}
                onClick={() => setFilter('rejected')}
              >
                Rejetés ({stats.rejected})
              </button>
            </div>

            <div className="search-box">
              <input
                type="text"
                placeholder="🔍 Rechercher..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="action-group">
              <button className="action-btn approve-all" onClick={approveAllPending}>
                ✓ Tout approuver
              </button>
              <button className="action-btn export-btn" onClick={exportDecisions}>
                📊 Exporter rapport
              </button>
              <button className="action-btn export-btn" onClick={exportCleanedDataset}>
                💾 Exporter dataset nettoyé
              </button>
            </div>
          </div>

          <div className="examples-list">
            {filteredExamples.map(example => (
              <ExampleCard
                key={example.id}
                example={example}
                onApprove={handleApprove}
                onReject={handleReject}
                onReset={handleReset}
              />
            ))}
          </div>

          {filteredExamples.length === 0 && (
            <div className="empty-state">
              <span className="empty-icon">🔍</span>
              <span>Aucun exemple trouvé</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
