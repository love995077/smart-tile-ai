from sqlalchemy import Column, Integer, String, Text, Numeric, DateTime, ForeignKey, text, JSON
from sqlalchemy.orm import relationship
from app.DB.database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    inventory_id = Column(String(255), unique=True, index=True)
    name = Column(Text, nullable=False)
    sku = Column(String(255), unique=True, index=True, nullable=False)
    slug = Column(Text, nullable=False)
    title = Column(Text)
    description = Column(Text)
    list_image = Column(Text)
    
    # JSON columns for application arrays and attributes
    application = Column(JSON)
    attributes = Column(JSON)
    
    category = Column(Integer)
    category_slug = Column(String(255), nullable=False)
    color = Column(Integer)
    sub_colors = Column(JSON)
    finish = Column(Integer)
    material = Column(Integer)
    size = Column(Integer)
    thickness = Column(Integer)
    
    price = Column(Integer)
    discount = Column(Integer)
    sale_price = Column(Integer)
    price_per_sqft = Column(Numeric(10, 2), default=0.00)
    piece_per_box = Column(Integer, default=0)
    
    short_description = Column(Text)
    long_description = Column(Text)
    unit = Column(String(255), nullable=False)
    stock_status = Column(String(55), default="in_stock")
    is_active = Column(Integer, default=1)
    status = Column(String(50), default="draft")
    
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'))

    # This creates a magic link to fetch all images for a product instantly!
    images = relationship("ProductImage", back_populates="product")


class ProductImage(Base):
    __tablename__ = "products_images"

    id = Column(Integer, primary_key=True, index=True)
    product_image = Column(Text)
    
    # This links directly to the 'id' column in the products table
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'))

    # The reverse link back to the parent product
    product = relationship("Product", back_populates="images")