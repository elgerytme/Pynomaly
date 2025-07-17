/**
 * React Provider component for Pynomaly SDK
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { PynomalyClient, PynomalyConfig } from '../../../index';
import { usePynomalyClient, UsePynomalyClientReturn } from '../hooks/usePynomalyClient';

export interface PynomalyProviderProps {
  children: ReactNode;
  config: PynomalyConfig;
  autoConnect?: boolean;
  onError?: (error: Error) => void;
  onReady?: (client: PynomalyClient) => void;
}

const PynomalyContext = createContext<UsePynomalyClientReturn | null>(null);

export const PynomalyProvider: React.FC<PynomalyProviderProps> = ({
  children,
  config,
  autoConnect = true,
  onError,
  onReady
}) => {
  const clientState = usePynomalyClient({
    ...config,
    autoConnect,
    onError,
    onReady
  });

  return (
    <PynomalyContext.Provider value={clientState}>
      {children}
    </PynomalyContext.Provider>
  );
};

export const usePynomaly = (): UsePynomalyClientReturn => {
  const context = useContext(PynomalyContext);
  if (!context) {
    throw new Error('usePynomaly must be used within a PynomalyProvider');
  }
  return context;
};

export default PynomalyProvider;