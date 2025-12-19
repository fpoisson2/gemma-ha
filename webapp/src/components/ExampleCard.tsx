import type { DatasetExample } from '../types';
import './ExampleCard.css';

interface ExampleCardProps {
  example: DatasetExample;
  onApprove: (id: number) => void;
  onReject: (id: number, reason?: string) => void;
  onReset: (id: number) => void;
}

const REJECTION_REASONS = [
  'Mauvaise action (turn_on au lieu de turn_off)',
  'Entité non pertinente',
  'Entité technique/système',
  'Commande incompréhensible',
  'Format incorrect',
  'Autre',
];

export function ExampleCard({ example, onApprove, onReject, onReset }: ExampleCardProps) {
  const { parsed, status, rejectionReason } = example;

  const getStatusClass = () => {
    switch (status) {
      case 'approved': return 'status-approved';
      case 'rejected': return 'status-rejected';
      default: return 'status-pending';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'approved': return '✓';
      case 'rejected': return '✗';
      default: return '?';
    }
  };

  const handleReject = (reason: string) => {
    onReject(example.id, reason);
  };

  return (
    <div className={`example-card ${getStatusClass()}`}>
      <div className="card-header">
        <span className="example-id">#{example.id}</span>
        <span className={`status-badge ${getStatusClass()}`}>
          {getStatusIcon()} {status}
        </span>
      </div>

      <div className="card-content">
        <div className="section user-section">
          <div className="section-label">
            <span className="icon">👤</span> User Query
          </div>
          <div className="section-content user-query">
            {parsed.userQuery}
          </div>
        </div>

        {parsed.entities.length > 0 && (
          <div className="section entities-section">
            <div className="section-label">
              <span className="icon">📋</span> Entités disponibles
            </div>
            <div className="entities-list">
              {parsed.entities.map((entityList, idx) => (
                <div key={idx} className="entity-domain">
                  <span className="domain-name">{entityList.domain}</span>
                  <div className="entity-tags">
                    {entityList.entities.map((entity, eidx) => (
                      <span
                        key={eidx}
                        className={`entity-tag ${isProblematicEntity(entity) ? 'problematic' : ''}`}
                        title={isProblematicEntity(entity) ? 'Entité potentiellement problématique' : ''}
                      >
                        {entity}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="section model-section">
          <div className="section-label">
            <span className="icon">🤖</span> Model Response
          </div>
          <div className="section-content model-response">
            {parsed.functionCall ? (
              <div className="function-call">
                <span className="function-action">{parsed.functionCall.action}</span>
                <span className="function-arrow">→</span>
                <span className="function-entity">{parsed.functionCall.entityId}</span>
              </div>
            ) : (
              <pre>{parsed.modelResponse}</pre>
            )}
          </div>
        </div>

        {status === 'rejected' && rejectionReason && (
          <div className="section rejection-section">
            <div className="section-label">
              <span className="icon">⚠️</span> Raison du rejet
            </div>
            <div className="rejection-reason">{rejectionReason}</div>
          </div>
        )}
      </div>

      <div className="card-actions">
        {status === 'pending' ? (
          <>
            <button className="btn btn-approve" onClick={() => onApprove(example.id)}>
              <span className="btn-icon">+</span> Approuver
            </button>
            <div className="reject-dropdown">
              <button className="btn btn-reject">
                <span className="btn-icon">−</span> Rejeter
              </button>
              <div className="dropdown-content">
                {REJECTION_REASONS.map((reason, idx) => (
                  <button key={idx} onClick={() => handleReject(reason)}>
                    {reason}
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <button className="btn btn-reset" onClick={() => onReset(example.id)}>
            ↺ Réinitialiser
          </button>
        )}
      </div>
    </div>
  );
}

function isProblematicEntity(entity: string): boolean {
  const problematicPatterns = [
    'zigbee2mqtt',
    'permit_join',
    'bridge',
    'disable_leds',
    'slzb',
    'gpio',
    'enable_outdoor',
    '_affichage',
    '_purificateur',
  ];
  return problematicPatterns.some(pattern =>
    entity.toLowerCase().includes(pattern.toLowerCase())
  );
}
