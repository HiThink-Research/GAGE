
import React from 'react';
import '../../assets/mahjong.scss';

// Dynamically require images or use paths
// React 16 + Webpack: usually require.context or direct imports.
// Simplified: assume they are copied to public or imported.
// Since we are inside src, we must import them. 
// But dynamic import of 34+ files is tedious.
// We can use a require context or just construct URL if in public.
// For CRA, importing is safest.

const importAll = (r) => {
  let images = {};
  r.keys().forEach((item, index) => { 
      const module = r(item);
      const key = item.replace('./', '').replace('.svg', '');
      images[key] = module.default || module; 
  });
  return images;
}

// Ensure this path matches where we downloaded: ../../assets/mahjong
// Warning: require.context is Webpack specific.
let svgs = {};
try {
    svgs = importAll(require.context('../../assets/mahjong', false, /\.svg$/));
} catch (e) {
    console.warn("Mahjong assets not found or require.context failed", e);
}

const MahjongTile = ({ tile, isHidden, isSelected, isSelectable, onClick, onDoubleClick, style }) => {
  // Map tile code to file name key
  // B1 -> B1 (if saved as B1.svg)
  let key = tile;
  if (isHidden) key = 'Back'; // We need a Back.svg or use a color
  
  const src = svgs[key] || svgs[key + '.svg'];

  return (
    <div 
      className={`mahjong-tile ${isSelected ? 'selected' : ''} ${isSelectable ? 'selectable' : ''}`} 
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      style={{
          width: 40, height: 56, 
          background: isHidden ? '#2c3e50' : '#f0f0f0',
          display: 'flex', justifyContent: 'center', alignItems: 'center',
          ...style
      }}
    >
      {src ? (
          <img src={src} alt={tile} style={{width: '100%', height: '100%'}} />
      ) : (
          // Fallback to text
          <span>{isHidden ? '🀫' : tile}</span>
      )}
    </div>
  );
};

export default MahjongTile;
