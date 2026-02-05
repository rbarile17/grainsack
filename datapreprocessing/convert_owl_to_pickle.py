"""Convert KG to complete pickle for faster loading.

Run this script to build complete KG objects and save them as kg_complete.pkl.
This includes all components: TSV data, RDF graph, entity types, and NetworkX graph.

Usage: python -m datapreprocessing.convert_owl_to_pickle [kg_name]
If no kg_name is provided, converts all KGs in the kgs directory.
"""

import pickle
import sys
from pathlib import Path

# Assuming this is run from the project root
KGS_PATH = Path("kgs")


def create_complete_kg_pickle(kg_name):
    """Create a complete KG pickle by building the full KG object."""
    complete_pickle_path = KGS_PATH / kg_name / "kg_complete.pkl"
    
    # Check if TSV files exist
    train_path = KGS_PATH / kg_name / "abox/splits/train.tsv"
    if not train_path.exists():
        print(f"❌ {train_path} not found, cannot create complete KG pickle")
        return False
    
    print(f"🏗️  Building complete KG object for {kg_name}...")
    
    # Import here to avoid circular imports
    try:
        from grainsack.kg import KG
    except ImportError:
        print(f"❌ Cannot import KG class. Make sure to run from project root")
        return False
    
    # Build the KG object (this will build from scratch if no pickle exists)
    kg = KG(kg_name)
    
    print(f"💾 Saving complete KG object to {complete_pickle_path}...")
    with open(complete_pickle_path, 'wb') as f:
        pickle.dump(kg, f)
    
    print(f"✅ Complete KG pickle created for {kg_name}")
    return True


def main():
    if len(sys.argv) > 1:
        # Convert specific KG
        kg_name = sys.argv[1]
        create_complete_kg_pickle(kg_name)
    else:
        # Convert all KGs
        if not KGS_PATH.exists():
            print(f"❌ {KGS_PATH} directory not found")
            sys.exit(1)
        
        kg_dirs = [d for d in KGS_PATH.iterdir() if d.is_dir() and (d / "abox/splits/train.tsv").exists()]
        
        if not kg_dirs:
            print(f"❌ No KGs with train.tsv files found in {KGS_PATH}")
            sys.exit(1)
        
        print(f"Found {len(kg_dirs)} KGs to convert\n")
        
        success_count = 0
        for kg_dir in kg_dirs:
            if create_complete_kg_pickle(kg_dir.name):
                success_count += 1
            print()
        
        print(f"Completed: {success_count}/{len(kg_dirs)} KGs converted successfully")


if __name__ == "__main__":
    main()
