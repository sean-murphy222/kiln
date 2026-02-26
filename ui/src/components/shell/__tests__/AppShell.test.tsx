import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, it, expect } from 'vitest';
import { NavRail } from '../NavRail';

describe('NavRail', () => {
  it('renders all four tool navigation items', () => {
    render(
      <MemoryRouter initialEntries={['/quarry']}>
        <NavRail />
      </MemoryRouter>,
    );

    expect(screen.getByText('Quarry')).toBeInTheDocument();
    expect(screen.getByText('Forge')).toBeInTheDocument();
    expect(screen.getByText('Foundry')).toBeInTheDocument();
    expect(screen.getByText('Hearth')).toBeInTheDocument();
  });

  it('renders settings link', () => {
    render(
      <MemoryRouter initialEntries={['/quarry']}>
        <NavRail />
      </MemoryRouter>,
    );

    expect(screen.getByLabelText('Settings')).toBeInTheDocument();
  });
});
